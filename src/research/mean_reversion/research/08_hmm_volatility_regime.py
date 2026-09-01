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
# PROJECT IMPORTS
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

RESEARCH_07_METADATA = CACHE_DIR / "research_07_event_metadata.csv"

RESEARCH_07_PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_STATE_CACHE = CACHE_DIR / "research_08_causal_hmm_states.csv"

HMM_DIAGNOSTIC = RESULTS_DIR / "research_08_hmm_window_diagnostic.csv"

STATE_Z_OUTPUT = RESULTS_DIR / "research_08_hmm_state_z_summary.csv"

BARRIER_OUTPUT = RESULTS_DIR / "research_08_hmm_state_barrier_summary.csv"

FAILURE_OUTPUT = RESULTS_DIR / "research_08_failure_analysis.csv"

WINDOW_OUTPUT = RESULTS_DIR / "research_08_hmm_window_summary.csv"


# =============================================================================
# RESEARCH PARAMETERS
# =============================================================================

Z_THRESHOLDS = (
    1.5,
    2.0,
    2.5,
    3.0,
)

HMM_STATES = (
    0,
    1,
    2,
)

SIDES = (
    "LONG",
    "SHORT",
)

HORIZONS = (
    10,
    20,
    30,
    60,
    120,
)

TARGETS = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
    75.0,
    100.0,
)

STOPS = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
)

MIN_HMM_TRAIN_OBSERVATIONS = 500

HMM_RANDOM_STATE = 42
HMM_N_ITER = 200


# =============================================================================
# HMM FEATURES
# MUST MATCH src/models/regime.py
# =============================================================================

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
# SAFE METRICS
# =============================================================================


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def safe_mean(values) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_median(values) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


# =============================================================================
# LOAD MARKET DATA
# =============================================================================


def load_market_data() -> pd.DataFrame:

    print("Loading MNQ data...")

    data = load_data()

    if data is None or data.empty:
        raise RuntimeError("MNQ data could not be loaded.")

    print(f"Rows loaded: {len(data):,}")

    return data.copy()


# =============================================================================
# PREPARE RTH
# =============================================================================


def prepare_rth(
    data: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Preparing RTH...")

    df = data.copy()

    if "timestamp ET" in df.columns:
        df["timestamp ET"] = pd.to_datetime(
            df["timestamp ET"],
            errors="coerce",
            utc=True,
        )

    elif "timestamp" in df.columns:
        df["timestamp ET"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

    else:
        raise KeyError("No timestamp column found.")

    df = df.dropna(subset=["timestamp ET"])

    df = df.sort_values("timestamp ET").reset_index(drop=True)

    local = df["timestamp ET"].dt.tz_convert("America/New_York")

    local_time = local.dt.time

    rth_start = pd.Timestamp("09:30:00").time()

    rth_end = pd.Timestamp("16:00:00").time()

    mask = (local_time >= rth_start) & (local_time < rth_end)

    rth = df.loc[mask].copy().reset_index(drop=True)

    print(f"RTH rows: {len(rth):,}")

    return rth


# =============================================================================
# BUILD FEATURES
# =============================================================================


def build_features(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Building complete feature set...")

    features = build_mean_reversion_features(rth.copy())

    features = features.reset_index(drop=True)

    features["timestamp ET"] = pd.to_datetime(
        features["timestamp ET"],
        errors="coerce",
        utc=True,
    )

    features = (
        features.dropna(subset=["timestamp ET"])
        .sort_values("timestamp ET")
        .reset_index(drop=True)
    )

    missing = [column for column in HMM_FEATURES if column not in features.columns]

    if missing:
        raise RuntimeError("Missing HMM features: " + ", ".join(missing))

    print(f"Feature columns: {len(features.columns)}")

    return features


# =============================================================================
# LOAD RESEARCH 07
# =============================================================================


def load_research_07():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not RESEARCH_07_METADATA.exists():
        raise FileNotFoundError(
            f"Missing Research 07 metadata:\n{RESEARCH_07_METADATA}"
        )

    if not RESEARCH_07_PATH_CACHE.exists():
        raise FileNotFoundError(
            f"Missing Research 07 path cache:\n{RESEARCH_07_PATH_CACHE}"
        )

    metadata = pd.read_csv(RESEARCH_07_METADATA)

    required = [
        "event_id",
        "data_index",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [column for column in required if column not in metadata.columns]

    if missing:
        raise RuntimeError("Research 07 metadata missing: " + ", ".join(missing))

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

    # CRITICAL:
    # Normalize every timestamp to UTC.
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

    cache_file = np.load(
        RESEARCH_07_PATH_CACHE,
        allow_pickle=False,
    )

    cache = {key: np.asarray(cache_file[key]) for key in cache_file.files}

    print(f"Metadata rows: {len(metadata):,}")

    print("Metadata columns:")

    print(metadata.columns.tolist())

    print()
    print("Cache arrays:")

    for key, value in cache.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    required_cache = [
        "future_close",
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    for key in required_cache:
        if key not in cache:
            raise RuntimeError(f"Research 07 cache missing '{key}'.")

        if len(cache[key]) != len(metadata):
            raise RuntimeError(
                f"Cache '{key}' length "
                f"{len(cache[key]):,} != metadata "
                f"length {len(metadata):,}."
            )

    return metadata, cache


# =============================================================================
# BUILD WINDOWS
# =============================================================================


def build_windows(
    metadata: pd.DataFrame,
) -> list[dict]:

    print()
    print("Building OOS windows from Research 07 metadata...")

    window_numbers = sorted(metadata["window"].unique().tolist())

    if not window_numbers:
        raise RuntimeError("No Research 07 windows found.")

    print(f"Research 07 OOS windows detected: {len(window_numbers)}")

    print(window_numbers)

    global_start = metadata["timestamp"].min()

    windows = []

    for number in window_numbers:
        subset = metadata.loc[metadata["window"] == number]

        oos_start = subset["timestamp"].min()

        oos_end = subset["timestamp"].max()

        windows.append(
            {
                "window": int(number),
                "train_start": global_start,
                "train_end": oos_start,
                "oos_start": oos_start,
                "oos_end": oos_end,
            }
        )

    print(f"Usable OOS windows: {len(windows)}")

    print(f"First OOS start: {windows[0]['oos_start']}")

    print(f"Last OOS end: {windows[-1]['oos_end']}")

    return windows


# =============================================================================
# EXACT TIMESTAMP HMM MAPPING
# =============================================================================


def build_causal_hmm_states(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    windows: list[dict],
) -> pd.DataFrame:

    section("BUILDING CAUSAL OOS HMM STATES")

    # -------------------------------------------------------------------------
    # IMPORTANT
    #
    # We DO NOT use Research 07 data_index here.
    #
    # Research 07 and the feature engine do not necessarily have identical
    # row indexing because preprocessing can remove rows.
    #
    # Timestamp is the canonical key.
    # -------------------------------------------------------------------------

    features = features.copy().sort_values("timestamp ET").reset_index(drop=True)

    # Exact timestamp lookup.
    #
    # One row per timestamp is expected for 1-minute RTH data.

    duplicate_timestamps = features["timestamp ET"].duplicated().sum()

    if duplicate_timestamps:
        raise RuntimeError(
            f"Features contain {duplicate_timestamps:,} duplicate timestamps."
        )

    state_records = []

    diagnostics = []

    for position, window in enumerate(
        windows,
        start=1,
    ):
        number = window["window"]

        oos_start = window["oos_start"]

        oos_end = window["oos_end"]

        # ---------------------------------------------------------------------
        # CAUSAL TRAINING SET
        #
        # Everything strictly before OOS start.
        # ---------------------------------------------------------------------

        train = features.loc[features["timestamp ET"] < oos_start].copy()

        # ---------------------------------------------------------------------
        # OOS FEATURE SET
        # ---------------------------------------------------------------------

        oos = features.loc[
            (features["timestamp ET"] >= oos_start)
            & (features["timestamp ET"] <= oos_end)
        ].copy()

        # ---------------------------------------------------------------------
        # Replicate prepare_data() validity check from regime.py.
        # ---------------------------------------------------------------------

        valid_train = (
            train[HMM_FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        valid_oos = (
            oos[HMM_FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        train_valid = len(valid_train)

        oos_valid = len(valid_oos)

        print(
            f"Window {position:02d}/"
            f"{len(windows)} | "
            f"TRAIN_ROWS={len(train):,} | "
            f"TRAIN_VALID={train_valid:,} | "
            f"OOS_ROWS={len(oos):,} | "
            f"OOS_VALID={oos_valid:,}"
        )

        # ---------------------------------------------------------------------
        # Insufficient history.
        # ---------------------------------------------------------------------

        if train_valid < MIN_HMM_TRAIN_OBSERVATIONS:
            diagnostics.append(
                {
                    "window": number,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "train_rows": len(train),
                    "train_valid": train_valid,
                    "oos_rows": len(oos),
                    "oos_valid": oos_valid,
                    "states_assigned": 0,
                    "status": "SKIPPED_INSUFFICIENT_TRAIN_HISTORY",
                }
            )

            print("  SKIPPED: insufficient causal HMM training history.")

            continue

        if oos_valid == 0:
            diagnostics.append(
                {
                    "window": number,
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

            print("  SKIPPED: no valid OOS observations.")

            continue

        # ---------------------------------------------------------------------
        # FIT EXACT PROJECT HMM
        # ---------------------------------------------------------------------

        model = VolatilityRegimeModel(
            n_states=3,
            random_state=HMM_RANDOM_STATE,
            n_iter=HMM_N_ITER,
        )

        model.fit(train)

        # ---------------------------------------------------------------------
        # PREDICT OOS
        #
        # predict_states() drops invalid rows internally.
        # Its returned index is the ORIGINAL dataframe index.
        # ---------------------------------------------------------------------

        predicted = model.predict_states(oos)

        states_assigned = len(predicted)

        # ---------------------------------------------------------------------
        # Save state keyed by exact timestamp.
        # ---------------------------------------------------------------------

        for feature_index, state in predicted.items():
            timestamp = oos.loc[
                feature_index,
                "timestamp ET",
            ]

            state_records.append(
                {
                    "timestamp": timestamp,
                    "window": number,
                    "hmm_state": int(state),
                }
            )

        diagnostics.append(
            {
                "window": number,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "train_rows": len(train),
                "train_valid": train_valid,
                "oos_rows": len(oos),
                "oos_valid": oos_valid,
                "states_assigned": states_assigned,
                "status": "OK",
            }
        )

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    diagnostic_df = pd.DataFrame(diagnostics)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_df.to_csv(
        HMM_DIAGNOSTIC,
        index=False,
    )

    # =========================================================================
    # STATE TABLE
    # =========================================================================

    states = pd.DataFrame(state_records)

    if states.empty:
        raise RuntimeError("No causal HMM states were generated.")

    states["timestamp"] = pd.to_datetime(
        states["timestamp"],
        errors="raise",
        utc=True,
    )

    states = states.sort_values("timestamp").reset_index(drop=True)

    # Check exact uniqueness.

    duplicate_state_keys = states["timestamp"].duplicated().sum()

    if duplicate_state_keys:
        raise RuntimeError(
            f"Generated HMM states contain "
            f"{duplicate_state_keys:,} duplicate timestamps."
        )

    # =========================================================================
    # MAP BY TIMESTAMP
    # =========================================================================

    print()
    print("MAPPING HMM STATES TO RESEARCH 07 EVENTS")

    events = metadata.copy()

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        errors="raise",
        utc=True,
    )

    events = events.merge(
        states[
            [
                "timestamp",
                "hmm_state",
            ]
        ],
        on="timestamp",
        how="left",
        validate="many_to_one",
    )

    events["hmm_state"] = events["hmm_state"].fillna(-1).astype(int)

    assigned = events["hmm_state"] >= 0

    assigned_count = int(assigned.sum())

    missing_count = int((~assigned).sum())

    print(f"Research 07 events: {len(events):,}")

    print(f"HMM states assigned: {assigned_count:,}")

    print(f"HMM states unavailable: {missing_count:,}")

    # -------------------------------------------------------------------------
    # Per-window coverage
    # -------------------------------------------------------------------------

    coverage = (
        events.groupby("window")
        .agg(
            events=(
                "event_id",
                "size",
            ),
            assigned=(
                "hmm_state",
                lambda x: int((x >= 0).sum()),
            ),
        )
        .reset_index()
    )

    coverage["coverage_pct"] = coverage["assigned"] / coverage["events"] * 100.0

    print()
    print("HMM coverage by window:")

    print(coverage.to_string(index=False))

    # =========================================================================
    # SAVE CACHE
    # =========================================================================

    events[
        [
            "event_id",
            "window",
            "timestamp",
            "hmm_state",
        ]
    ].to_csv(
        HMM_STATE_CACHE,
        index=False,
    )

    print()
    print("HMM state cache saved:")

    print(HMM_STATE_CACHE)

    return events


# =============================================================================
# LOAD VALID HMM CACHE
# =============================================================================


def try_load_hmm_cache(
    metadata: pd.DataFrame,
):

    if not HMM_STATE_CACHE.exists():
        return None

    print()
    print("Checking existing HMM state cache...")

    try:
        cached = pd.read_csv(HMM_STATE_CACHE)

        required = [
            "event_id",
            "window",
            "timestamp",
            "hmm_state",
        ]

        if any(column not in cached.columns for column in required):
            print("Existing HMM cache is incompatible.")

            return None

        cached["event_id"] = pd.to_numeric(
            cached["event_id"],
            errors="raise",
        ).astype(np.int64)

        cached["window"] = pd.to_numeric(
            cached["window"],
            errors="raise",
        ).astype(np.int64)

        cached["hmm_state"] = pd.to_numeric(
            cached["hmm_state"],
            errors="raise",
        ).astype(int)

        cached["timestamp"] = pd.to_datetime(
            cached["timestamp"],
            errors="raise",
            utc=True,
        )

        # Compare event IDs.

        metadata_ids = set(metadata["event_id"])

        cached_ids = set(cached["event_id"])

        if metadata_ids != cached_ids:
            print("Existing HMM cache does not match Research 07 events.")

            return None

        # Compare timestamp mapping.

        merged = metadata[
            [
                "event_id",
                "timestamp",
            ]
        ].merge(
            cached[
                [
                    "event_id",
                    "timestamp",
                ]
            ],
            on="event_id",
            how="inner",
            suffixes=(
                "_metadata",
                "_cache",
            ),
            validate="one_to_one",
        )

        if not (merged["timestamp_metadata"] == merged["timestamp_cache"]).all():
            print("Existing HMM cache timestamp mapping is incompatible.")

            return None

        print(f"Valid HMM cache found: {len(cached):,} events.")

        return metadata.merge(
            cached[
                [
                    "event_id",
                    "hmm_state",
                ]
            ],
            on="event_id",
            how="left",
            validate="one_to_one",
        )

    except Exception as exc:
        print(f"Existing HMM cache rejected: {exc}")

        return None


# =============================================================================
# GET EVENTS
# =============================================================================


def get_events_with_hmm(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    windows: list[dict],
) -> pd.DataFrame:

    cached = try_load_hmm_cache(metadata)

    if cached is not None:
        print("Using existing HMM state cache.")

        return cached

    return build_causal_hmm_states(
        features,
        metadata,
        windows,
    )


# =============================================================================
# EVENT MASK
# =============================================================================


def get_event_mask(
    events: pd.DataFrame,
    side: str,
    z: float,
) -> np.ndarray:

    zscore = events["zscore_30"].to_numpy(dtype=float)

    if side == "LONG":
        return zscore <= -z

    if side == "SHORT":
        return zscore >= z

    raise ValueError(f"Unknown side: {side}")


# =============================================================================
# PATH EVALUATION
# =============================================================================


def evaluate_paths(
    favorable: np.ndarray,
    adverse: np.ndarray,
    stop: float,
    target: float,
    horizon: int,
) -> pd.DataFrame:

    horizon = min(
        horizon,
        favorable.shape[1],
    )

    favorable = favorable[
        :,
        :horizon,
    ]

    adverse = adverse[
        :,
        :horizon,
    ]

    n = len(favorable)

    target_hit = favorable >= target

    stop_hit = adverse >= stop

    target_time = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
    )

    stop_time = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
    )

    target_any = target_hit.any(axis=1)

    stop_any = stop_hit.any(axis=1)

    if target_any.any():
        target_time[target_any] = (
            np.argmax(
                target_hit[target_any],
                axis=1,
            )
            + 1
        )

    if stop_any.any():
        stop_time[stop_any] = (
            np.argmax(
                stop_hit[stop_any],
                axis=1,
            )
            + 1
        )

    # Conservative:
    # same candle TP + SL = STOP.

    target_first = target_time < stop_time

    stop_first = stop_time <= target_time

    exit_type = np.full(
        n,
        "TIMEOUT",
        dtype=object,
    )

    points = np.zeros(
        n,
        dtype=np.float32,
    )

    exit_bar = np.full(
        n,
        horizon,
        dtype=np.int16,
    )

    exit_type[target_first] = "TARGET"

    points[target_first] = target

    exit_bar[target_first] = target_time[target_first]

    exit_type[stop_first] = "STOP"

    points[stop_first] = -stop

    exit_bar[stop_first] = stop_time[stop_first]

    return pd.DataFrame(
        {
            "exit_type": exit_type,
            "R_points": points,
            "exit_bar": exit_bar,
        }
    )


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(
    outcome: pd.DataFrame,
) -> dict:

    n = len(outcome)

    if n == 0:
        return {
            "observations": 0,
            "wins": 0,
            "stops": 0,
            "timeouts": 0,
            "win_rate": np.nan,
            "stop_rate": np.nan,
            "timeout_rate": np.nan,
            "total_points": 0.0,
            "mean_points": np.nan,
            "profit_factor": np.nan,
            "mean_holding_bars": np.nan,
        }

    wins = int((outcome["exit_type"] == "TARGET").sum())

    stops = int((outcome["exit_type"] == "STOP").sum())

    timeouts = int((outcome["exit_type"] == "TIMEOUT").sum())

    points = outcome["R_points"].to_numpy(dtype=float)

    gross_profit = points[points > 0].sum()

    gross_loss = -points[points < 0].sum()

    return {
        "observations": n,
        "wins": wins,
        "stops": stops,
        "timeouts": timeouts,
        "win_rate": safe_ratio(
            wins,
            n,
        ),
        "stop_rate": safe_ratio(
            stops,
            n,
        ),
        "timeout_rate": safe_ratio(
            timeouts,
            n,
        ),
        "total_points": float(points.sum()),
        "mean_points": float(points.mean()),
        "profit_factor": safe_ratio(
            gross_profit,
            gross_loss,
        ),
        "mean_holding_bars": float(outcome["exit_bar"].mean()),
    }


# =============================================================================
# STATE × Z
# =============================================================================


def run_state_z_analysis(
    events: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    section("RUNNING HMM STATE × Z-SCORE ANALYSIS")

    usable = events["hmm_state"] >= 0

    events = events.loc[usable].reset_index(drop=True)

    rows = []

    for side in SIDES:
        if side == "LONG":
            favorable = cache["long_favorable"]

            adverse = cache["long_adverse"]

        else:
            favorable = cache["short_favorable"]

            adverse = cache["short_adverse"]

        for state in HMM_STATES:
            for z in Z_THRESHOLDS:
                mask = get_event_mask(
                    events,
                    side,
                    z,
                ) & (events["hmm_state"].to_numpy() == state)

                ids = events.loc[
                    mask,
                    "event_id",
                ].to_numpy(dtype=np.int64)

                if len(ids) == 0:
                    continue

                outcome = evaluate_paths(
                    favorable[ids],
                    adverse[ids],
                    stop=20.0,
                    target=20.0,
                    horizon=120,
                )

                metrics = calculate_metrics(outcome)

                rows.append(
                    {
                        "side": side,
                        "hmm_state": state,
                        "zscore_threshold": z,
                        "stop_points": 20.0,
                        "target_points": 20.0,
                        "horizon_bars": 120,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# FULL BARRIER GRID
# =============================================================================


def run_barrier_grid(
    events: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    section("RUNNING HMM STATE × SL × TP × HORIZON")

    events = events.loc[events["hmm_state"] >= 0].reset_index(drop=True)

    rows = []

    total_experiments = (
        len(SIDES)
        * len(HMM_STATES)
        * len(Z_THRESHOLDS)
        * len(HORIZONS)
        * len(STOPS)
        * len(TARGETS)
    )

    experiment = 0

    for side in SIDES:
        if side == "LONG":
            favorable = cache["long_favorable"]

            adverse = cache["long_adverse"]

        else:
            favorable = cache["short_favorable"]

            adverse = cache["short_adverse"]

        for state in HMM_STATES:
            for z in Z_THRESHOLDS:
                mask = get_event_mask(
                    events,
                    side,
                    z,
                ) & (events["hmm_state"].to_numpy() == state)

                ids = events.loc[
                    mask,
                    "event_id",
                ].to_numpy(dtype=np.int64)

                if len(ids) == 0:
                    continue

                f = favorable[ids]

                a = adverse[ids]

                for horizon in HORIZONS:
                    for stop in STOPS:
                        for target in TARGETS:
                            experiment += 1

                            outcome = evaluate_paths(
                                f,
                                a,
                                stop,
                                target,
                                horizon,
                            )

                            metrics = calculate_metrics(outcome)

                            rows.append(
                                {
                                    "side": side,
                                    "hmm_state": state,
                                    "zscore_threshold": z,
                                    "stop_points": stop,
                                    "target_points": target,
                                    "horizon_bars": horizon,
                                    **metrics,
                                }
                            )

                            if experiment == 1 or experiment % 500 == 0:
                                print(
                                    f"  experiment {experiment:,}/{total_experiments:,}"
                                )

    return pd.DataFrame(rows)


# =============================================================================
# FAILURE ANALYSIS
# =============================================================================


def run_failure_analysis(
    events: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    section("RUNNING FAILURE ANALYSIS")

    events = events.loc[events["hmm_state"] >= 0].reset_index(drop=True)

    rows = []

    for side in SIDES:
        if side == "LONG":
            favorable = cache["long_favorable"]

            adverse = cache["long_adverse"]

        else:
            favorable = cache["short_favorable"]

            adverse = cache["short_adverse"]

        for state in HMM_STATES:
            for z in Z_THRESHOLDS:
                mask = get_event_mask(
                    events,
                    side,
                    z,
                ) & (events["hmm_state"].to_numpy() == state)

                ids = events.loc[
                    mask,
                    "event_id",
                ].to_numpy(dtype=np.int64)

                if len(ids) == 0:
                    continue

                f = favorable[
                    ids,
                    :120,
                ]

                a = adverse[
                    ids,
                    :120,
                ]

                mfe = np.max(
                    f,
                    axis=1,
                )

                mae = np.max(
                    a,
                    axis=1,
                )

                categories = np.full(
                    len(ids),
                    "NO_EDGE",
                    dtype=object,
                )

                # Both large favorable and adverse movement.
                categories[((mfe >= 20.0) & (mae >= 20.0))] = "FAVORABLE_THEN_ADVERSE"

                # Large move in expected direction.
                categories[mfe >= 50.0] = "LARGE_FAVORABLE_MOVE"

                # Strong continuation against the mean-reversion trade.
                categories[((mae >= 30.0) & (mfe < 10.0))] = "TREND_CONTINUATION"

                # Neither side moves much.
                categories[((mfe < 10.0) & (mae < 10.0))] = "NO_FOLLOW_THROUGH"

                total = len(categories)

                for category in sorted(set(categories)):
                    category_mask = categories == category

                    count = int(category_mask.sum())

                    rows.append(
                        {
                            "side": side,
                            "hmm_state": state,
                            "zscore_threshold": z,
                            "category": category,
                            "count": count,
                            "rate": safe_ratio(
                                count,
                                total,
                            ),
                            "observations": total,
                            "mean_mfe": safe_mean(mfe[category_mask]),
                            "median_mfe": safe_median(mfe[category_mask]),
                            "mean_mae": safe_mean(mae[category_mask]),
                            "median_mae": safe_median(mae[category_mask]),
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# WINDOW COVERAGE
# =============================================================================


def build_window_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:

    return (
        events.assign(hmm_available=(events["hmm_state"] >= 0))
        .groupby("window")
        .agg(
            events=(
                "event_id",
                "size",
            ),
            hmm_available=(
                "hmm_available",
                "sum",
            ),
        )
        .reset_index()
        .assign(hmm_coverage_pct=lambda x: x["hmm_available"] / x["events"] * 100.0)
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    section("MEAN REVERSION — RESEARCH 08")

    print("HMM VOLATILITY REGIME ANALYSIS")

    print("-" * 100)

    print("No final strategy.")

    print("No parameter optimization.")

    print("No XGBoost.")

    print("No production changes.")

    print("Research 07 path cache reused.")

    # =========================================================================
    # DATA
    # =========================================================================

    data = load_market_data()

    rth = prepare_rth(data)

    features = build_features(rth)

    # =========================================================================
    # RESEARCH 07
    # =========================================================================

    metadata, cache = load_research_07()

    # =========================================================================
    # WINDOWS
    # =========================================================================

    windows = build_windows(metadata)

    # =========================================================================
    # HMM STATES
    # =========================================================================

    events = get_events_with_hmm(
        features,
        metadata,
        windows,
    )

    # =========================================================================
    # VALIDATE COVERAGE BEFORE EXPENSIVE GRID
    # =========================================================================

    section("VALIDATING HMM EVENT COVERAGE")

    coverage = build_window_summary(events)

    print(coverage.to_string(index=False))

    usable_events = int((events["hmm_state"] >= 0).sum())

    if usable_events == 0:
        raise RuntimeError("Zero Research 07 events have a valid causal HMM state.")

    print()
    print(f"Usable events for HMM analysis: {usable_events:,}/{len(events):,}")

    # =========================================================================
    # STATE × Z
    # =========================================================================

    state_z = run_state_z_analysis(
        events,
        cache,
    )

    # =========================================================================
    # FULL GRID
    # =========================================================================

    barrier = run_barrier_grid(
        events,
        cache,
    )

    # =========================================================================
    # FAILURE
    # =========================================================================

    failure = run_failure_analysis(
        events,
        cache,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_z.to_csv(
        STATE_Z_OUTPUT,
        index=False,
    )

    barrier.to_csv(
        BARRIER_OUTPUT,
        index=False,
    )

    failure.to_csv(
        FAILURE_OUTPUT,
        index=False,
    )

    coverage.to_csv(
        WINDOW_OUTPUT,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    section("RESEARCH 08 COMPLETE")

    print(f"Research 07 events: {len(events):,}")

    print(f"Events with causal HMM state: {usable_events:,}")

    print(f"Events without causal HMM state: {len(events) - usable_events:,}")

    print()
    print("FILES SAVED")

    print(HMM_STATE_CACHE)

    print(HMM_DIAGNOSTIC)

    print(STATE_Z_OUTPUT)

    print(BARRIER_OUTPUT)

    print(FAILURE_OUTPUT)

    print(WINDOW_OUTPUT)

    # =========================================================================
    # TOP RESULTS
    # =========================================================================

    if not barrier.empty:
        section("TOP DESCRIPTIVE COMBINATIONS")

        top = (
            barrier.query("observations >= 500")
            .sort_values(
                [
                    "profit_factor",
                    "mean_points",
                ],
                ascending=False,
            )
            .head(30)
        )

        columns = [
            "side",
            "hmm_state",
            "zscore_threshold",
            "stop_points",
            "target_points",
            "horizon_bars",
            "observations",
            "win_rate",
            "stop_rate",
            "timeout_rate",
            "mean_points",
            "profit_factor",
        ]

        print(top[columns].to_string(index=False))

    section("RESEARCH INTEGRITY")

    print("HMM states remain raw labels: 0 / 1 / 2.")

    print("HMM states are NOT interpreted as volatility labels.")

    print("The HMM implementation is exactly src/models/regime.py.")

    print("Training is strictly causal.")

    print("Research 07 event-to-HMM mapping uses exact UTC timestamps.")

    print("Research 07 path cache is reused.")

    print("Same-bar TARGET + STOP -> STOP.")

    print("No final strategy was constructed.")


if __name__ == "__main__":
    main()
