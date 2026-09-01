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

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

R07_METADATA = CACHE_DIR / "research_07_event_metadata.csv"

R07_PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

# NEW CACHE.
# Never overwrite the old broken Research 08 cache.
NEW_HMM_CACHE = CACHE_DIR / "research_08b_causal_hmm_states.csv"

DIAGNOSTIC_OUTPUT = RESULTS_DIR / "research_08b_hmm_coverage_diagnostic.csv"

WINDOW_OUTPUT = RESULTS_DIR / "research_08b_hmm_window_coverage.csv"


# =============================================================================
# HMM CONFIGURATION
# EXACT PROJECT MODEL
# =============================================================================

N_STATES = 3
RANDOM_STATE = 42
N_ITER = 200

MIN_TRAIN_VALID = 500

HMM_FEATURES = [
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
]


# =============================================================================
# DISPLAY
# =============================================================================


def section(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# LOAD RESEARCH 07
# =============================================================================


def load_research_07():

    section("LOADING RESEARCH 07")

    if not R07_METADATA.exists():
        raise FileNotFoundError(f"Research 07 metadata not found:\n{R07_METADATA}")

    if not R07_PATH_CACHE.exists():
        raise FileNotFoundError(f"Research 07 path cache not found:\n{R07_PATH_CACHE}")

    metadata = pd.read_csv(R07_METADATA)

    required = [
        "event_id",
        "data_index",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in metadata.columns]

    if missing:
        raise RuntimeError(
            "Research 07 metadata missing columns: " + ", ".join(missing)
        )

    metadata["event_id"] = pd.to_numeric(
        metadata["event_id"],
        errors="raise",
    ).astype(np.int64)

    metadata["data_index"] = pd.to_numeric(
        metadata["data_index"],
        errors="raise",
    ).astype(np.int64)

    metadata["window"] = pd.to_numeric(
        metadata["window"],
        errors="raise",
    ).astype(np.int64)

    metadata["zscore_30"] = pd.to_numeric(
        metadata["zscore_30"],
        errors="coerce",
    )

    # Canonical timestamp.
    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"],
        errors="coerce",
        utc=True,
    )

    metadata = (
        metadata.dropna(
            subset=[
                "timestamp",
                "zscore_30",
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Load path cache ONLY to verify integrity.
    # We do not rebuild it.
    # -------------------------------------------------------------------------

    path_cache = np.load(
        R07_PATH_CACHE,
        allow_pickle=False,
    )

    cache = {key: np.asarray(path_cache[key]) for key in path_cache.files}

    print(f"Research 07 events: {len(metadata):,}")

    print(f"Research 07 windows: {metadata['window'].nunique()}")

    print(f"Research 07 first timestamp: {metadata['timestamp'].min()}")

    print(f"Research 07 last timestamp: {metadata['timestamp'].max()}")

    print()

    for key, value in cache.items():
        print(f"{key}: shape={value.shape}")

    # Hard integrity check.

    if len(cache["future_close"]) != len(metadata):
        raise RuntimeError(
            "Research 07 path cache and metadata have different event counts."
        )

    return metadata


# =============================================================================
# LOAD RAW DATA
# =============================================================================


def load_market_data():

    section("LOADING MARKET DATA")

    data = load_data()

    if data is None or data.empty:
        raise RuntimeError("load_data() returned no data.")

    data = data.copy()

    print(f"Rows loaded: {len(data):,}")

    return data


# =============================================================================
# NORMALIZE RAW TIMESTAMP
# =============================================================================


def normalize_raw_timestamp(
    data: pd.DataFrame,
) -> pd.DataFrame:

    df = data.copy()

    if "timestamp ET" in df.columns:
        timestamp_column = "timestamp ET"

    elif "timestamp" in df.columns:
        timestamp_column = "timestamp"

    else:
        raise KeyError("Raw data has no timestamp column.")

    df["canonical_timestamp"] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    df = (
        df.dropna(subset=["canonical_timestamp"])
        .sort_values("canonical_timestamp")
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# BUILD FEATURES
# =============================================================================


def build_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    section("BUILDING FEATURES")

    # IMPORTANT:
    #
    # We intentionally do NOT reproduce the old RTH filtering here.
    #
    # The feature engine itself is the canonical producer of the HMM features.
    # This prevents Research 08 from silently changing the universe before
    # the feature engine gets a chance to construct its rolling variables.

    features = build_mean_reversion_features(raw.copy())

    if features is None or features.empty:
        raise RuntimeError("Feature engine returned no rows.")

    # -------------------------------------------------------------------------
    # Find timestamp.
    # -------------------------------------------------------------------------

    if "timestamp ET" in features.columns:
        source_timestamp = "timestamp ET"

    elif "timestamp" in features.columns:
        source_timestamp = "timestamp"

    else:
        raise KeyError("Feature dataframe has no timestamp column.")

    features["canonical_timestamp"] = pd.to_datetime(
        features[source_timestamp],
        errors="coerce",
        utc=True,
    )

    features = (
        features.dropna(subset=["canonical_timestamp"])
        .sort_values("canonical_timestamp")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Required HMM columns.
    # -------------------------------------------------------------------------

    missing = [feature for feature in HMM_FEATURES if feature not in features.columns]

    if missing:
        raise RuntimeError(
            "Feature engine does not contain required HMM "
            "features: " + ", ".join(missing)
        )

    # -------------------------------------------------------------------------
    # Duplicate timestamp protection.
    # -------------------------------------------------------------------------

    duplicate_count = int(features["canonical_timestamp"].duplicated().sum())

    if duplicate_count:
        raise RuntimeError(
            f"Feature dataframe contains {duplicate_count:,} duplicate timestamps."
        )

    print(f"Feature rows: {len(features):,}")

    print(f"Feature first timestamp: {features['canonical_timestamp'].min()}")

    print(f"Feature last timestamp: {features['canonical_timestamp'].max()}")

    print(f"Feature columns: {len(features.columns)}")

    return features


# =============================================================================
# CAUSAL HMM
# =============================================================================


def calculate_valid_hmm_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    valid = (
        dataframe[HMM_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    return valid


def build_causal_states(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    section("BUILDING CLEAN CAUSAL HMM STATES")

    windows = sorted(metadata["window"].unique().tolist())

    print(f"Windows detected: {len(windows)}")

    state_parts = []

    diagnostics = []

    for position, window_number in enumerate(
        windows,
        start=1,
    ):
        events = metadata.loc[metadata["window"] == window_number].copy()

        oos_start = events["timestamp"].min()

        oos_end = events["timestamp"].max()

        # =====================================================================
        # STRICT CAUSAL TRAINING
        #
        # Everything before the FIRST timestamp of the OOS window.
        # =====================================================================

        train = features.loc[features["canonical_timestamp"] < oos_start].copy()

        oos = features.loc[
            (features["canonical_timestamp"] >= oos_start)
            & (features["canonical_timestamp"] <= oos_end)
        ].copy()

        valid_train = calculate_valid_hmm_rows(train)

        valid_oos = calculate_valid_hmm_rows(oos)

        train_valid = len(valid_train)

        oos_valid = len(valid_oos)

        print()
        print(f"Window {position:02d}/{len(windows)}")

        print(f"  OOS: {oos_start} -> {oos_end}")

        print(f"  TRAIN rows: {len(train):,}")

        print(f"  TRAIN valid: {train_valid:,}")

        print(f"  OOS rows: {len(oos):,}")

        print(f"  OOS valid: {oos_valid:,}")

        # =====================================================================
        # WINDOW 1 / WARMUP
        # =====================================================================

        if train_valid < MIN_TRAIN_VALID:
            print("  STATUS: SKIPPED — insufficient causal training history.")

            diagnostics.append(
                {
                    "window": window_number,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "train_rows": len(train),
                    "train_valid": train_valid,
                    "oos_rows": len(oos),
                    "oos_valid": oos_valid,
                    "states_assigned": 0,
                    "status": "SKIPPED_INSUFFICIENT_HISTORY",
                }
            )

            continue

        if oos_valid == 0:
            print("  STATUS: SKIPPED — zero valid HMM observations.")

            diagnostics.append(
                {
                    "window": window_number,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "train_rows": len(train),
                    "train_valid": train_valid,
                    "oos_rows": len(oos),
                    "oos_valid": oos_valid,
                    "states_assigned": 0,
                    "status": "SKIPPED_NO_VALID_OOS",
                }
            )

            continue

        # =====================================================================
        # FIT EXACT PROJECT MODEL
        # =====================================================================

        model = VolatilityRegimeModel(
            n_states=N_STATES,
            random_state=RANDOM_STATE,
            n_iter=N_ITER,
        )

        model.fit(train)

        # =====================================================================
        # PREDICT ONLY VALID OOS ROWS
        # =====================================================================

        oos_states = model.predict_states(oos)

        if oos_states.empty:
            raise RuntimeError(f"Window {window_number}: HMM produced zero states.")

        # predict_states returns Series indexed by the original oos index.

        timestamps = oos.loc[
            oos_states.index,
            "canonical_timestamp",
        ].to_numpy()

        part = pd.DataFrame(
            {
                "timestamp": timestamps,
                "window": window_number,
                "hmm_state": oos_states.to_numpy(dtype=np.int8),
            }
        )

        state_parts.append(part)

        diagnostics.append(
            {
                "window": window_number,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "train_rows": len(train),
                "train_valid": train_valid,
                "oos_rows": len(oos),
                "oos_valid": oos_valid,
                "states_assigned": len(part),
                "status": "OK",
            }
        )

        print(f"  STATUS: OK — states assigned {len(part):,}")

    # =========================================================================
    # CONCATENATE
    # =========================================================================

    if not state_parts:
        raise RuntimeError("No HMM states were generated.")

    states = pd.concat(
        state_parts,
        ignore_index=True,
    )

    states["timestamp"] = pd.to_datetime(
        states["timestamp"],
        errors="raise",
        utc=True,
    )

    states["window"] = states["window"].astype(np.int64)

    states["hmm_state"] = states["hmm_state"].astype(np.int8)

    states = states.sort_values(
        [
            "window",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # =========================================================================
    # CHECK STATE TIMESTAMP UNIQUENESS
    # =========================================================================

    duplicate_states = int(
        states[
            [
                "window",
                "timestamp",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_states:
        raise RuntimeError(
            f"Generated HMM state table contains "
            f"{duplicate_states:,} duplicate "
            f"(window,timestamp) pairs."
        )

    # =========================================================================
    # MAP BACK TO RESEARCH 07
    # =========================================================================

    print()
    print("MAPPING STATES TO RESEARCH 07 EVENTS")

    event_map = metadata[
        [
            "event_id",
            "window",
            "timestamp",
        ]
    ].copy()

    event_map["timestamp"] = pd.to_datetime(
        event_map["timestamp"],
        errors="raise",
        utc=True,
    )

    mapped = event_map.merge(
        states,
        on=[
            "window",
            "timestamp",
        ],
        how="left",
        validate="one_to_one",
    )

    mapped_count = int(mapped["hmm_state"].notna().sum())

    missing_count = int(mapped["hmm_state"].isna().sum())

    print()
    print(f"Research 07 events: {len(mapped):,}")

    print(f"Mapped HMM states: {mapped_count:,}")

    print(f"Missing HMM states: {missing_count:,}")

    # =========================================================================
    # WINDOW DIAGNOSTIC
    # =========================================================================

    coverage = (
        mapped.assign(assigned=lambda x: x["hmm_state"].notna())
        .groupby("window")
        .agg(
            events=(
                "event_id",
                "size",
            ),
            assigned=(
                "assigned",
                "sum",
            ),
        )
        .reset_index()
    )

    coverage["coverage_pct"] = coverage["assigned"] / coverage["events"] * 100.0

    print()
    print("COVERAGE BY WINDOW")

    print(coverage.to_string(index=False))

    # =========================================================================
    # SAVE DIAGNOSTICS
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_df = pd.DataFrame(diagnostics)

    diagnostic_df.to_csv(
        DIAGNOSTIC_OUTPUT,
        index=False,
    )

    coverage.to_csv(
        WINDOW_OUTPUT,
        index=False,
    )

    # =========================================================================
    # CRITICAL INTEGRITY RULE
    #
    # Do NOT silently accept missing states.
    #
    # Window 1 can be legitimately unavailable because there is no training
    # history. Every later Research 07 event should have a state if its HMM
    # feature vector is available.
    # =========================================================================

    later_windows = coverage.loc[coverage["window"] > coverage["window"].min()]

    bad_later_windows = later_windows.loc[later_windows["coverage_pct"] < 99.0]

    if not bad_later_windows.empty:
        print()
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        print("HMM COVERAGE FAILURE")

        print("The script will STOP before any strategy/grid analysis.")

        print(bad_later_windows.to_string(index=False))

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        raise RuntimeError(
            "Causal HMM coverage is incomplete in later "
            "Research 07 windows. No research grid was executed."
        )

    # =========================================================================
    # ALL VALID
    # =========================================================================

    mapped["hmm_state"] = mapped["hmm_state"].astype(int)

    invalid = set(mapped["hmm_state"].unique()) - {0, 1, 2}

    if invalid:
        raise RuntimeError(f"Unexpected HMM states: {invalid}")

    # =========================================================================
    # SAVE CLEAN CACHE
    # =========================================================================

    mapped[
        [
            "event_id",
            "window",
            "timestamp",
            "hmm_state",
        ]
    ].to_csv(
        NEW_HMM_CACHE,
        index=False,
    )

    return mapped, diagnostic_df


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("RESEARCH 08B — CLEAN HMM ENGINE")

    print("Purpose:")

    print("Rebuild causal HMM states independently of the broken Research 08 cache.")

    print()
    print("IMPORTANT:")

    print("No SL/TP optimization.")

    print("No strategy construction.")

    print("No volatility-regime filtering.")

    print("No HMM state interpretation.")

    print("Only causal HMM state construction + integrity validation.")

    # =========================================================================
    # LOAD RESEARCH 07
    # =========================================================================

    metadata = load_research_07()

    # =========================================================================
    # LOAD RAW
    # =========================================================================

    raw = load_market_data()

    # =========================================================================
    # BUILD FEATURES
    # =========================================================================

    features = build_features(raw)

    # =========================================================================
    # RANGE CHECK BEFORE HMM
    # =========================================================================

    section("TEMPORAL RANGE CHECK")

    r07_start = metadata["timestamp"].min()

    r07_end = metadata["timestamp"].max()

    feature_start = features["canonical_timestamp"].min()

    feature_end = features["canonical_timestamp"].max()

    print(f"Research 07: {r07_start} -> {r07_end}")

    print(f"Features:    {feature_start} -> {feature_end}")

    # =========================================================================
    # CRITICAL:
    #
    # If features don't reach Research 07's final timestamp, STOP NOW.
    # Do NOT waste time fitting 22 HMMs.
    # =========================================================================

    if feature_end < r07_end:
        print()
        print("FEATURE COVERAGE FAILURE")

        print(f"Research 07 ends at {r07_end}")

        print(f"Feature data ends at {feature_end}")

        print()
        print("NO HMM TRAINING WAS PERFORMED.")

        raise RuntimeError(
            "Feature engine does not cover the complete "
            "Research 07 temporal range. "
            "This must be fixed before HMM analysis."
        )

    # =========================================================================
    # BUILD STATES
    # =========================================================================

    events, diagnostics = build_causal_states(
        features,
        metadata,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    section("RESEARCH 08B COMPLETE")

    print(f"Events: {len(events):,}")

    print("Causal HMM state cache:")

    print(NEW_HMM_CACHE)

    print()
    print("Diagnostics:")

    print(DIAGNOSTIC_OUTPUT)

    print(WINDOW_OUTPUT)

    print()
    print("NO STRATEGY GRID WAS RUN.")

    print("NO PARAMETERS WERE OPTIMIZED.")

    print("HMM states remain raw labels 0 / 1 / 2.")


if __name__ == "__main__":
    main()
