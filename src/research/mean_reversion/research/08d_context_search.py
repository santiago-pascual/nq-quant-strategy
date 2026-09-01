from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# IMPORTS
# =============================================================================

from src.databento_loader import load_databento_mnq
from src.data_validator import validate_dataset
from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT_CONTEXT = RESULTS_DIR / "research_08d_context_summary.csv"

OUTPUT_DETAIL = RESULTS_DIR / "research_08d_context_detail.csv"

OUTPUT_WINDOW = RESULTS_DIR / "research_08d_context_window_summary.csv"

OUTPUT_VOL = RESULTS_DIR / "research_08d_volatility_distribution.csv"


# =============================================================================
# RESEARCH DEFINITIONS
# =============================================================================

HMM_STATES = (
    0,
    1,
    2,
)

Z_THRESHOLDS = (
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
)

HORIZONS = (
    10,
    20,
    30,
    60,
    120,
)

VOL_BUCKETS = (
    "0-20",
    "20-40",
    "40-60",
    "60-80",
    "80-100",
)

MIN_EVENTS = 100
MIN_WINDOW_EVENTS = 20


# =============================================================================
# PRINT
# =============================================================================


def section(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# LOAD RESEARCH 07 METADATA
# =============================================================================


def load_metadata() -> pd.DataFrame:

    section("LOADING RESEARCH 07 METADATA")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Research 07 metadata not found:\n{METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    required = [
        "event_id",
        "data_index",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing Research 07 columns: " + ", ".join(missing))

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["data_index"] = pd.to_numeric(
        df["data_index"],
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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    df = df.sort_values("event_id").reset_index(drop=True)

    print(f"Events: {len(df):,}")

    print(f"Windows: {df['window'].nunique()}")

    print(f"First timestamp: {df['timestamp'].min()}")

    print(f"Last timestamp:  {df['timestamp'].max()}")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm() -> pd.DataFrame:

    section("LOADING RESEARCH 08B HMM CACHE")

    if not HMM_PATH.exists():
        raise FileNotFoundError(f"Research 08B HMM cache not found:\n{HMM_PATH}")

    hmm = pd.read_csv(
        HMM_PATH,
        usecols=[
            "event_id",
            "hmm_state",
        ],
    )

    hmm["event_id"] = pd.to_numeric(
        hmm["event_id"],
        errors="raise",
    ).astype(np.int64)

    hmm["hmm_state"] = pd.to_numeric(
        hmm["hmm_state"],
        errors="raise",
    ).astype(np.int8)

    if hmm["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event IDs in HMM cache.")

    print(f"HMM events: {len(hmm):,}")

    print("State counts:")

    print(hmm["hmm_state"].value_counts().sort_index().to_string())

    return hmm


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_path_cache():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Research 07 path cache not found:\n{PATH_CACHE_PATH}")

    archive = np.load(
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
        if key not in archive.files:
            raise RuntimeError(f"Missing path-cache array: {key}")

    paths = {
        key: np.asarray(
            archive[key],
            dtype=np.float32,
        )
        for key in required
    }

    print(f"Cached events: {paths['future_close'].shape[0]:,}")

    print(f"Maximum horizon: {paths['future_close'].shape[1]}")

    return paths


# =============================================================================
# LOAD MARKET + BUILD VOLATILITY
# =============================================================================


def load_project_mnq() -> pd.DataFrame:

    section("LOADING MNQ DATA")

    print("Using project loader:")

    print("src.databento_loader.load_databento_mnq()")

    df = load_databento_mnq()

    print(f"Rows loaded: {len(df):,}")

    validate_dataset(df)

    # -------------------------------------------------------------------------
    # TIMESTAMP
    # -------------------------------------------------------------------------

    if "timestamp ET" not in df.columns:
        raise RuntimeError("Dataset does not contain 'timestamp ET'.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp ET"],
        utc=True,
        errors="raise",
    )

    # -------------------------------------------------------------------------
    # RETURN FEATURES
    # -------------------------------------------------------------------------

    print("Building return features...")

    df = add_return_features(df)

    # -------------------------------------------------------------------------
    # VOLATILITY FEATURES
    # -------------------------------------------------------------------------

    print("Building volatility features...")

    df = add_volatility_features(df)

    # -------------------------------------------------------------------------
    # CRITICAL VALIDATION
    # -------------------------------------------------------------------------

    if "realized_vol_30" not in df.columns:
        raise RuntimeError(
            "\n"
            "realized_vol_30 was NOT created by "
            "add_volatility_features().\n"
            "\n"
            "Available volatility columns:\n"
            + "\n".join(
                [c for c in df.columns if "vol" in c.lower() or "variance" in c.lower()]
            )
        )

    valid_vol = df["realized_vol_30"].notna().sum()

    print(f"Feature rows: {len(df):,}")

    print(f"Feature columns: {len(df.columns)}")

    print(f"Valid realized_vol_30: {valid_vol:,}")

    print(f"Missing realized_vol_30: {df['realized_vol_30'].isna().sum():,}")

    if valid_vol == 0:
        raise RuntimeError("realized_vol_30 contains no valid observations.")

    print(f"First timestamp: {df['timestamp'].min()}")

    print(f"Last timestamp:  {df['timestamp'].max()}")

    return df.sort_values("timestamp").reset_index(drop=True)


# =============================================================================
# CAUSAL VOLATILITY PERCENTILE
# =============================================================================


def build_causal_volatility_percentile(
    metadata: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:

    section("BUILDING CAUSAL VOLATILITY PERCENTILE")

    # =========================================================================
    # IMPORTANT DESIGN CHOICE
    # =========================================================================
    #
    # Research 07 already contains `data_index`.
    #
    # That is the exact row position in the original MNQ dataset.
    #
    # Therefore we DO NOT use timestamps to map volatility.
    #
    # We directly attach realized_vol_30 using data_index.
    #
    # This avoids all timestamp timezone / precision / alignment problems.
    # =========================================================================

    if "data_index" not in metadata.columns:
        raise RuntimeError("Research 07 metadata does not contain data_index.")

    if "realized_vol_30" not in market.columns:
        raise RuntimeError("Market data does not contain realized_vol_30.")

    # =========================================================================
    # PREPARE MARKET VOLATILITY
    # =========================================================================

    market_vol = market[
        [
            "realized_vol_30",
        ]
    ].copy()

    market_vol["realized_vol_30"] = pd.to_numeric(
        market_vol["realized_vol_30"],
        errors="coerce",
    )

    market_values = np.array(
        market_vol["realized_vol_30"].to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
    )

    print(f"Market rows: {len(market_values):,}")

    print("Mapping Research 07 events through data_index...")

    # =========================================================================
    # VALIDATE INDICES
    # =========================================================================

    data_indices = pd.to_numeric(
        metadata["data_index"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    invalid_low = data_indices < 0

    invalid_high = data_indices >= len(market_values)

    invalid = invalid_low | invalid_high

    if invalid.any():
        raise RuntimeError(
            f"{invalid.sum():,} Research 07 events have invalid data_index values."
        )

    # =========================================================================
    # DIRECT VOLATILITY MAPPING
    # =========================================================================

    event_volatility = np.array(
        market_values[data_indices],
        dtype=np.float64,
        copy=True,
    )

    events = metadata[
        [
            "event_id",
            "data_index",
            "window",
            "timestamp",
        ]
    ].copy()

    events["realized_vol_30"] = event_volatility

    # =========================================================================
    # COVERAGE
    # =========================================================================

    valid_events = np.isfinite(event_volatility)

    print(f"Events mapped to realized_vol_30: {valid_events.sum():,}/{len(events):,}")

    if not valid_events.any():
        raise RuntimeError("No Research 07 events have valid realized_vol_30 values.")

    # =========================================================================
    # CAUSAL PERCENTILE
    # =========================================================================
    #
    # For each Research 07 OOS window:
    #
    # TRAIN DISTRIBUTION =
    # all MARKET volatility observations whose
    # data_index is strictly before the first event
    # of that OOS window.
    #
    # This is causal.
    #
    # No future OOS volatility is used to define the regime.
    # =========================================================================

    events["vol_percentile"] = np.nan

    windows = sorted(events["window"].unique())

    print(f"Research 07 windows: {len(windows)}")

    for position, window_id in enumerate(
        windows,
        start=1,
    ):
        window_mask = events["window"] == window_id

        if not window_mask.any():
            continue

        # ---------------------------------------------------------------------
        # FIRST OOS DATA INDEX
        # ---------------------------------------------------------------------

        oos_start_index = int(
            events.loc[
                window_mask,
                "data_index",
            ].min()
        )

        # ---------------------------------------------------------------------
        # CAUSAL TRAIN VOLATILITY
        # ---------------------------------------------------------------------
        #
        # Strictly BEFORE OOS start.
        # ---------------------------------------------------------------------

        train_values = np.array(
            market_values[:oos_start_index],
            dtype=np.float64,
            copy=True,
        )

        train_values = train_values[np.isfinite(train_values)]

        if len(train_values) < 100:
            print(
                f"Window {position:02d}/{len(windows)} "
                f"| TRAIN volatility={len(train_values):,} "
                f"| insufficient"
            )

            continue

        # Ensure writable.
        train_values = np.array(
            train_values,
            dtype=np.float64,
            copy=True,
        )

        train_values.sort()

        # ---------------------------------------------------------------------
        # OOS VALUES
        # ---------------------------------------------------------------------

        oos_values = np.array(
            events.loc[
                window_mask,
                "realized_vol_30",
            ].to_numpy(dtype=np.float64),
            dtype=np.float64,
            copy=True,
        )

        valid_oos = np.isfinite(oos_values)

        percentiles = np.full(
            len(oos_values),
            np.nan,
            dtype=np.float64,
        )

        if valid_oos.any():
            percentiles[valid_oos] = (
                np.searchsorted(
                    train_values,
                    oos_values[valid_oos],
                    side="right",
                )
                / len(train_values)
                * 100.0
            )

        events.loc[
            window_mask,
            "vol_percentile",
        ] = percentiles

        print(
            f"Window {position:02d}/{len(windows)} "
            f"| OOS={window_mask.sum():,} "
            f"| TRAIN volatility={len(train_values):,} "
            f"| assigned={valid_oos.sum():,}"
        )

    # =========================================================================
    # VOLATILITY BUCKETS
    # =========================================================================

    def assign_bucket(value):

        if not np.isfinite(value):
            return "UNKNOWN"

        if value < 20.0:
            return "0-20"

        if value < 40.0:
            return "20-40"

        if value < 60.0:
            return "40-60"

        if value < 80.0:
            return "60-80"

        return "80-100"

    events["vol_bucket"] = events["vol_percentile"].map(assign_bucket)

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    assigned = events["vol_percentile"].notna().sum()

    unknown = (events["vol_bucket"] == "UNKNOWN").sum()

    print()

    print("VOLATILITY MAPPING VALIDATION")

    print(f"Total events: {len(events):,}")

    print(f"Valid volatility: {valid_events.sum():,}")

    print(f"Percentile assigned: {assigned:,}")

    print(f"Unknown bucket: {unknown:,}")

    if assigned == 0:
        raise RuntimeError("No volatility percentiles were assigned.")

    return (
        events[
            [
                "event_id",
                "realized_vol_30",
                "vol_percentile",
                "vol_bucket",
            ]
        ]
        .sort_values("event_id")
        .reset_index(drop=True)
    )


# =============================================================================
# FINAL EVENT TABLE
# =============================================================================


def build_event_table(
    metadata,
    hmm,
    volatility,
):

    section("BUILDING FINAL EVENT TABLE")

    events = metadata.merge(
        hmm,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    events = events.merge(
        volatility[
            [
                "event_id",
                "realized_vol_30",
                "vol_percentile",
                "vol_bucket",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    missing_hmm = events["hmm_state"].isna().sum()

    missing_vol = events["vol_bucket"].eq("UNKNOWN").sum()

    print(f"Events: {len(events):,}")

    print(f"Missing HMM states: {missing_hmm:,}")

    print(f"Missing volatility regimes: {missing_vol:,}")

    if missing_hmm:
        raise RuntimeError("HMM event coverage incomplete.")

    if missing_vol > len(events) * 0.05:
        raise RuntimeError(
            f"More than 5% of events have no volatility regime: {missing_vol:,}"
        )

    return events


# =============================================================================
# PATH ANALYSIS
# =============================================================================


def evaluate_path(
    favorable,
    adverse,
    target,
    stop,
    horizon,
):

    horizon = min(
        horizon,
        favorable.shape[1],
    )

    f = favorable[
        :,
        :horizon,
    ]

    a = adverse[
        :,
        :horizon,
    ]

    target_hit = f >= target

    stop_hit = a >= stop

    target_exists = target_hit.any(axis=1)

    stop_exists = stop_hit.any(axis=1)

    target_time = np.where(
        target_exists,
        np.argmax(
            target_hit,
            axis=1,
        ),
        horizon + 1,
    )

    stop_time = np.where(
        stop_exists,
        np.argmax(
            stop_hit,
            axis=1,
        ),
        horizon + 1,
    )

    win = target_time < stop_time

    loss = stop_time <= target_time

    timeout = ~target_exists & ~stop_exists

    n = len(f)

    return {
        "n": n,
        "win_rate": (float(win.mean()) if n else np.nan),
        "loss_rate": (float(loss.mean()) if n else np.nan),
        "timeout_rate": (float(timeout.mean()) if n else np.nan),
    }


# =============================================================================
# CONTEXT ANALYSIS
# =============================================================================


def analyze_contexts(
    events,
    paths,
):

    section("RUNNING CONTEXT ANALYSIS")

    z = events["zscore_30"].to_numpy(dtype=np.float32)

    hmm = events["hmm_state"].to_numpy(dtype=np.int8)

    vol = events["vol_bucket"].to_numpy()

    window = events["window"].to_numpy(dtype=np.int16)

    rows = []

    diagnostic_pairs = (
        (5.0, 5.0),
        (10.0, 5.0),
        (15.0, 10.0),
        (20.0, 10.0),
        (30.0, 15.0),
    )

    for side in (
        "LONG",
        "SHORT",
    ):
        if side == "LONG":
            favorable = paths["long_favorable"]

            adverse = paths["long_adverse"]

        else:
            favorable = paths["short_favorable"]

            adverse = paths["short_adverse"]

        for state in HMM_STATES:
            state_mask = hmm == state

            for bucket in VOL_BUCKETS:
                vol_mask = vol == bucket

                for threshold in Z_THRESHOLDS:
                    if side == "LONG":
                        z_mask = z <= -threshold

                    else:
                        z_mask = z >= threshold

                    mask = state_mask & vol_mask & z_mask

                    ids = np.flatnonzero(mask)

                    n = len(ids)

                    if n < MIN_EVENTS:
                        continue

                    unique_windows = np.unique(window[ids])

                    row = {
                        "side": side,
                        "hmm_state": state,
                        "vol_bucket": bucket,
                        "zscore": threshold,
                        "observations": n,
                        "windows": len(unique_windows),
                    }

                    for horizon in HORIZONS:
                        for target, stop in diagnostic_pairs:
                            result = evaluate_path(
                                favorable[ids],
                                adverse[ids],
                                target,
                                stop,
                                horizon,
                            )

                            name = f"wr_{int(target)}_before_{int(stop)}_h{horizon}"

                            row[name] = result["win_rate"]

                    rows.append(row)

    result = pd.DataFrame(rows)

    print(f"Context rows generated: {len(result):,}")

    return result


# =============================================================================
# SUMMARY
# =============================================================================


def summarize_contexts(
    detail,
):

    section("SUMMARIZING CONTEXTS")

    if detail.empty:
        return detail.copy()

    wr_columns = [c for c in detail.columns if c.startswith("wr_")]

    summary = detail.copy()

    summary["mean_path_wr"] = summary[wr_columns].mean(axis=1)

    summary["median_path_wr"] = summary[wr_columns].median(axis=1)

    summary["paths_wr_ge_50"] = summary[wr_columns].ge(0.50).sum(axis=1)

    summary["paths_wr_ge_55"] = summary[wr_columns].ge(0.55).sum(axis=1)

    summary["paths_wr_ge_60"] = summary[wr_columns].ge(0.60).sum(axis=1)

    summary = summary.sort_values(
        [
            "paths_wr_ge_60",
            "paths_wr_ge_55",
            "paths_wr_ge_50",
            "mean_path_wr",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return summary


# =============================================================================
# VOLATILITY DISTRIBUTION
# =============================================================================


def build_volatility_distribution(
    events,
):

    return events.groupby(
        "vol_bucket",
        as_index=False,
    ).agg(
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
        min_realized_vol_30=(
            "realized_vol_30",
            "min",
        ),
        max_realized_vol_30=(
            "realized_vol_30",
            "max",
        ),
        mean_percentile=(
            "vol_percentile",
            "mean",
        ),
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08D")

    print("CONTEXT DISCOVERY")

    print("-" * 100)

    print("LONG / SHORT")

    print("HMM state 0 / 1 / 2")

    print("Volatility percentile:")

    print("0-20 / 20-40 / 40-60 / 60-80 / 80-100")

    print("Z-score:")

    print(Z_THRESHOLDS)

    print()

    print("No HMM retraining.")

    print("No SL/TP optimization.")

    print("No final strategy.")

    # -------------------------------------------------------------------------
    # EXISTING RESEARCH DATA
    # -------------------------------------------------------------------------

    metadata = load_metadata()

    hmm = load_hmm()

    paths = load_path_cache()

    # -------------------------------------------------------------------------
    # MARKET
    # -------------------------------------------------------------------------

    market = load_project_mnq()

    # -------------------------------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------------------------------

    volatility = build_causal_volatility_percentile(
        metadata,
        market,
    )

    # -------------------------------------------------------------------------
    # FINAL EVENTS
    # -------------------------------------------------------------------------

    events = build_event_table(
        metadata,
        hmm,
        volatility,
    )

    # -------------------------------------------------------------------------
    # VOL DISTRIBUTION
    # -------------------------------------------------------------------------

    vol_distribution = build_volatility_distribution(events)

    print()

    print("VOLATILITY BUCKET DISTRIBUTION")

    print(vol_distribution.to_string(index=False))

    # -------------------------------------------------------------------------
    # CONTEXT SEARCH
    # -------------------------------------------------------------------------

    detail = analyze_contexts(
        events,
        paths,
    )

    summary = summarize_contexts(detail)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CONTEXT,
        index=False,
    )

    detail.to_csv(
        OUTPUT_DETAIL,
        index=False,
    )

    vol_distribution.to_csv(
        OUTPUT_VOL,
        index=False,
    )

    # -------------------------------------------------------------------------
    # TOP RESULTS
    # -------------------------------------------------------------------------

    section("TOP CONTEXTS")

    if summary.empty:
        print("No valid contexts found.")

    else:
        cols = [
            "side",
            "hmm_state",
            "vol_bucket",
            "zscore",
            "observations",
            "windows",
            "mean_path_wr",
            "median_path_wr",
            "paths_wr_ge_50",
            "paths_wr_ge_55",
            "paths_wr_ge_60",
        ]

        print(summary[cols].head(50).to_string(index=False))

    # -------------------------------------------------------------------------
    # HIGH QUALITY CONTEXTS
    # -------------------------------------------------------------------------

    section("CONTEXTS WITH >= 3 PATH RELATIONSHIPS ABOVE 50%")

    if summary.empty:
        print("(none)")

    else:
        high = summary[summary["paths_wr_ge_50"] >= 3]

        if high.empty:
            print("None.")

        else:
            print(
                high[
                    [
                        "side",
                        "hmm_state",
                        "vol_bucket",
                        "zscore",
                        "observations",
                        "windows",
                        "mean_path_wr",
                        "median_path_wr",
                        "paths_wr_ge_50",
                        "paths_wr_ge_55",
                        "paths_wr_ge_60",
                    ]
                ]
                .head(50)
                .to_string(index=False)
            )

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    section("RESEARCH 08D COMPLETE")

    print(f"Events analyzed: {len(events):,}")

    print()

    print("FILES SAVED")

    print(OUTPUT_CONTEXT)

    print(OUTPUT_DETAIL)

    print(OUTPUT_VOL)

    print()

    print("Volatility:")

    print("realized_vol_30 from src.feature_engine")

    print()

    print("Regimes:")

    print("Causal percentile: 0-20 / 20-40 / 40-60 / 60-80 / 80-100")

    print()

    print("HMM:")

    print("Raw causal states from Research 08B")

    print()

    print("No SL/TP optimization.")

    print("No final strategy.")


if __name__ == "__main__":
    main()
