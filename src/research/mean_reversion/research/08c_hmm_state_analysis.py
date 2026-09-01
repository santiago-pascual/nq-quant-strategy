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
# PATHS
# =============================================================================

RESULTS_DIR = ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

R07_METADATA = CACHE_DIR / "research_07_event_metadata.csv"

R07_PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_STATE_CACHE = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT_STATE = RESULTS_DIR / "research_08c_state_summary.csv"

OUTPUT_STATE_Z = RESULTS_DIR / "research_08c_state_zscore_summary.csv"

OUTPUT_TRANSITIONS = RESULTS_DIR / "research_08c_state_transitions.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08c_state_window_summary.csv"

OUTPUT_DISTRIBUTION = RESULTS_DIR / "research_08c_state_distribution.csv"


# =============================================================================
# PARAMETERS
# =============================================================================

STATES = (0, 1, 2)

SIDES = (
    "LONG",
    "SHORT",
)

Z_THRESHOLDS = (
    1.5,
    2.0,
    2.5,
    3.0,
)

HORIZONS = (
    5,
    10,
    20,
    30,
    60,
    120,
)

# These are descriptive levels only.
# They are NOT strategy optimization parameters.
MOVE_LEVELS = (
    5.0,
    10.0,
    20.0,
    35.0,
    50.0,
    75.0,
    100.0,
)


# =============================================================================
# DISPLAY
# =============================================================================


def section(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# SAFE FUNCTIONS
# =============================================================================


def safe_mean(values) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_median(values) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def safe_std(values) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


# =============================================================================
# LOAD METADATA
# =============================================================================


def load_metadata() -> pd.DataFrame:

    section("LOADING RESEARCH 07 METADATA")

    if not R07_METADATA.exists():
        raise FileNotFoundError(f"Missing Research 07 metadata:\n{R07_METADATA}")

    metadata = pd.read_csv(R07_METADATA)

    required = [
        "event_id",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in metadata.columns]

    if missing:
        raise RuntimeError("Missing metadata columns: " + ", ".join(missing))

    metadata["event_id"] = pd.to_numeric(
        metadata["event_id"],
        errors="raise",
    ).astype(np.int64)

    metadata["window"] = pd.to_numeric(
        metadata["window"],
        errors="raise",
    ).astype(np.int64)

    metadata["close"] = pd.to_numeric(
        metadata["close"],
        errors="coerce",
    )

    metadata["zscore_30"] = pd.to_numeric(
        metadata["zscore_30"],
        errors="coerce",
    )

    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"],
        errors="raise",
        utc=True,
    )

    metadata = metadata.sort_values("event_id").reset_index(drop=True)

    print(f"Events: {len(metadata):,}")

    print(f"Windows: {metadata['window'].nunique()}")

    return metadata


# =============================================================================
# LOAD HMM STATES
# =============================================================================


def load_hmm_states(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    section("LOADING CLEAN HMM STATE CACHE")

    if not HMM_STATE_CACHE.exists():
        raise FileNotFoundError(
            f"Missing clean HMM cache:\n{HMM_STATE_CACHE}\n\nRun 08b first."
        )

    states = pd.read_csv(HMM_STATE_CACHE)

    required = [
        "event_id",
        "window",
        "timestamp",
        "hmm_state",
    ]

    missing = [c for c in required if c not in states.columns]

    if missing:
        raise RuntimeError("HMM cache missing columns: " + ", ".join(missing))

    states["event_id"] = pd.to_numeric(
        states["event_id"],
        errors="raise",
    ).astype(np.int64)

    states["window"] = pd.to_numeric(
        states["window"],
        errors="raise",
    ).astype(np.int64)

    states["hmm_state"] = pd.to_numeric(
        states["hmm_state"],
        errors="raise",
    ).astype(int)

    states["timestamp"] = pd.to_datetime(
        states["timestamp"],
        errors="raise",
        utc=True,
    )

    # -------------------------------------------------------------------------
    # Exact integrity checks.
    # -------------------------------------------------------------------------

    if len(states) != len(metadata):
        raise RuntimeError(
            f"HMM state count {len(states):,} != Research 07 events {len(metadata):,}."
        )

    if set(states["event_id"]) != set(metadata["event_id"]):
        raise RuntimeError("HMM cache event IDs do not exactly match Research 07.")

    merged_check = metadata[
        [
            "event_id",
            "window",
            "timestamp",
        ]
    ].merge(
        states[
            [
                "event_id",
                "window",
                "timestamp",
            ]
        ],
        on="event_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_metadata",
            "_hmm",
        ),
    )

    if not (merged_check["window_metadata"] == merged_check["window_hmm"]).all():
        raise RuntimeError("HMM cache window mapping is inconsistent.")

    if not (merged_check["timestamp_metadata"] == merged_check["timestamp_hmm"]).all():
        raise RuntimeError("HMM cache timestamp mapping is inconsistent.")

    invalid_states = set(states["hmm_state"].unique()) - set(STATES)

    if invalid_states:
        raise RuntimeError(f"Unexpected HMM states: {invalid_states}")

    print(f"HMM states: {len(states):,}")

    print("State counts:")

    print(states["hmm_state"].value_counts().sort_index().to_string())

    return states


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_path_cache():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not R07_PATH_CACHE.exists():
        raise FileNotFoundError(f"Missing path cache:\n{R07_PATH_CACHE}")

    archive = np.load(
        R07_PATH_CACHE,
        allow_pickle=False,
    )

    cache = {key: np.asarray(archive[key]) for key in archive.files}

    required = [
        "future_close",
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    for key in required:
        if key not in cache:
            raise RuntimeError(f"Missing path cache array: {key}")

    n_events = len(cache["future_close"])

    if n_events != 825717:
        print(f"WARNING: path cache events = {n_events:,}")

    for key in required:
        if len(cache[key]) != n_events:
            raise RuntimeError(f"Array length mismatch: {key}")

    print(f"Cached events: {n_events:,}")

    print(f"Maximum horizon: {cache['future_close'].shape[1]}")

    return cache


# =============================================================================
# MERGE EVENT DATA
# =============================================================================


def build_events(
    metadata: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:

    section("BUILDING EVENT TABLE")

    events = metadata.merge(
        states[
            [
                "event_id",
                "hmm_state",
            ]
        ],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    if len(events) != len(metadata):
        raise RuntimeError("Event/state merge lost observations.")

    events = events.sort_values("event_id").reset_index(drop=True)

    print(f"Final events: {len(events):,}")

    return events


# =============================================================================
# BASIC STATE SUMMARY
# =============================================================================


def run_state_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:

    section("STATE DISTRIBUTION")

    rows = []

    total = len(events)

    for state in STATES:
        subset = events.loc[events["hmm_state"] == state]

        count = len(subset)

        rows.append(
            {
                "hmm_state": state,
                "observations": count,
                "share": safe_ratio(
                    count,
                    total,
                ),
                "mean_zscore": safe_mean(subset["zscore_30"]),
                "median_zscore": safe_median(subset["zscore_30"]),
                "mean_abs_zscore": safe_mean(np.abs(subset["zscore_30"])),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# STATE × Z × SIDE × HORIZON
# =============================================================================


def run_state_z_analysis(
    events: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    section("STATE × Z-SCORE × SIDE × HORIZON")

    rows = []

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    states = events["hmm_state"].to_numpy(dtype=int)

    zscores = events["zscore_30"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # We calculate directional movement from the cached favorable/adverse
    # paths, not from candle closes.
    # -------------------------------------------------------------------------

    for side in SIDES:
        if side == "LONG":
            favorable = cache["long_favorable"]

            adverse = cache["long_adverse"]

        else:
            favorable = cache["short_favorable"]

            adverse = cache["short_adverse"]

        for state in STATES:
            state_mask = states == state

            for z in Z_THRESHOLDS:
                if side == "LONG":
                    z_mask = zscores <= -z

                else:
                    z_mask = zscores >= z

                mask = state_mask & z_mask

                ids = event_ids[mask]

                if len(ids) == 0:
                    continue

                for horizon in HORIZONS:
                    h = min(
                        horizon,
                        favorable.shape[1],
                    )

                    f = favorable[
                        ids,
                        h - 1,
                    ]

                    a = adverse[
                        ids,
                        h - 1,
                    ]

                    # Path MFE / MAE through the horizon.
                    mfe = np.max(
                        favorable[
                            ids,
                            :h,
                        ],
                        axis=1,
                    )

                    mae = np.max(
                        adverse[
                            ids,
                            :h,
                        ],
                        axis=1,
                    )

                    close_move = cache["future_close"][
                        ids,
                        h - 1,
                    ]

                    rows.append(
                        {
                            "side": side,
                            "hmm_state": state,
                            "zscore_threshold": z,
                            "horizon_bars": horizon,
                            "observations": len(ids),
                            "mean_close_move": safe_mean(close_move),
                            "median_close_move": safe_median(close_move),
                            "std_close_move": safe_std(close_move),
                            "mean_mfe": safe_mean(mfe),
                            "median_mfe": safe_median(mfe),
                            "mean_mae": safe_mean(mae),
                            "median_mae": safe_median(mae),
                            "mfe_ge_5": safe_ratio(
                                np.sum(mfe >= 5),
                                len(mfe),
                            ),
                            "mfe_ge_10": safe_ratio(
                                np.sum(mfe >= 10),
                                len(mfe),
                            ),
                            "mfe_ge_20": safe_ratio(
                                np.sum(mfe >= 20),
                                len(mfe),
                            ),
                            "mfe_ge_35": safe_ratio(
                                np.sum(mfe >= 35),
                                len(mfe),
                            ),
                            "mfe_ge_50": safe_ratio(
                                np.sum(mfe >= 50),
                                len(mfe),
                            ),
                            "mfe_ge_75": safe_ratio(
                                np.sum(mfe >= 75),
                                len(mfe),
                            ),
                            "mfe_ge_100": safe_ratio(
                                np.sum(mfe >= 100),
                                len(mfe),
                            ),
                            "mae_ge_5": safe_ratio(
                                np.sum(mae >= 5),
                                len(mae),
                            ),
                            "mae_ge_10": safe_ratio(
                                np.sum(mae >= 10),
                                len(mae),
                            ),
                            "mae_ge_20": safe_ratio(
                                np.sum(mae >= 20),
                                len(mae),
                            ),
                            "mae_ge_35": safe_ratio(
                                np.sum(mae >= 35),
                                len(mae),
                            ),
                            "mae_ge_50": safe_ratio(
                                np.sum(mae >= 50),
                                len(mae),
                            ),
                            "mae_ge_75": safe_ratio(
                                np.sum(mae >= 75),
                                len(mae),
                            ),
                            "mae_ge_100": safe_ratio(
                                np.sum(mae >= 100),
                                len(mae),
                            ),
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# STATE-ONLY PATH ANALYSIS
# =============================================================================


def run_state_path_summary(
    events: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    section("STATE-ONLY PATH ANALYSIS")

    rows = []

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    states = events["hmm_state"].to_numpy(dtype=int)

    for side in SIDES:
        if side == "LONG":
            favorable = cache["long_favorable"]

            adverse = cache["long_adverse"]

        else:
            favorable = cache["short_favorable"]

            adverse = cache["short_adverse"]

        for state in STATES:
            mask = states == state

            ids = event_ids[mask]

            if len(ids) == 0:
                continue

            for horizon in HORIZONS:
                h = min(
                    horizon,
                    favorable.shape[1],
                )

                f_path = favorable[
                    ids,
                    :h,
                ]

                a_path = adverse[
                    ids,
                    :h,
                ]

                mfe = np.max(
                    f_path,
                    axis=1,
                )

                mae = np.max(
                    a_path,
                    axis=1,
                )

                rows.append(
                    {
                        "side": side,
                        "hmm_state": state,
                        "horizon_bars": horizon,
                        "observations": len(ids),
                        "mean_mfe": safe_mean(mfe),
                        "median_mfe": safe_median(mfe),
                        "mean_mae": safe_mean(mae),
                        "median_mae": safe_median(mae),
                        "mfe_mae_ratio": safe_ratio(
                            safe_mean(mfe),
                            safe_mean(mae),
                        ),
                        "prob_mfe_ge_5": safe_ratio(
                            np.sum(mfe >= 5),
                            len(mfe),
                        ),
                        "prob_mfe_ge_10": safe_ratio(
                            np.sum(mfe >= 10),
                            len(mfe),
                        ),
                        "prob_mfe_ge_20": safe_ratio(
                            np.sum(mfe >= 20),
                            len(mfe),
                        ),
                        "prob_mfe_ge_35": safe_ratio(
                            np.sum(mfe >= 35),
                            len(mfe),
                        ),
                        "prob_mfe_ge_50": safe_ratio(
                            np.sum(mfe >= 50),
                            len(mfe),
                        ),
                        "prob_mfe_ge_75": safe_ratio(
                            np.sum(mfe >= 75),
                            len(mfe),
                        ),
                        "prob_mfe_ge_100": safe_ratio(
                            np.sum(mfe >= 100),
                            len(mfe),
                        ),
                        "prob_mae_ge_5": safe_ratio(
                            np.sum(mae >= 5),
                            len(mae),
                        ),
                        "prob_mae_ge_10": safe_ratio(
                            np.sum(mae >= 10),
                            len(mae),
                        ),
                        "prob_mae_ge_20": safe_ratio(
                            np.sum(mae >= 20),
                            len(mae),
                        ),
                        "prob_mae_ge_35": safe_ratio(
                            np.sum(mae >= 35),
                            len(mae),
                        ),
                        "prob_mae_ge_50": safe_ratio(
                            np.sum(mae >= 50),
                            len(mae),
                        ),
                        "prob_mae_ge_75": safe_ratio(
                            np.sum(mae >= 75),
                            len(mae),
                        ),
                        "prob_mae_ge_100": safe_ratio(
                            np.sum(mae >= 100),
                            len(mae),
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# STATE TRANSITIONS
# =============================================================================


def run_transition_analysis(
    events: pd.DataFrame,
) -> pd.DataFrame:

    section("HMM STATE TRANSITION ANALYSIS")

    states = events["hmm_state"].to_numpy(dtype=int)

    if len(states) < 2:
        return pd.DataFrame()

    previous = states[:-1]

    current = states[1:]

    matrix = np.zeros(
        (
            len(STATES),
            len(STATES),
        ),
        dtype=np.int64,
    )

    for p, c in zip(
        previous,
        current,
    ):
        if p in STATES and c in STATES:
            matrix[
                p,
                c,
            ] += 1

    rows = []

    for state_from in STATES:
        total = matrix[state_from].sum()

        for state_to in STATES:
            count = matrix[
                state_from,
                state_to,
            ]

            rows.append(
                {
                    "from_state": state_from,
                    "to_state": state_to,
                    "count": int(count),
                    "probability": safe_ratio(
                        count,
                        total,
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# WINDOW STABILITY
# =============================================================================


def run_window_analysis(
    events: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    section("WINDOW-BY-WINDOW STABILITY")

    rows = []

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    states = events["hmm_state"].to_numpy(dtype=int)

    windows = sorted(events["window"].unique().tolist())

    for window in windows:
        window_mask = events["window"].to_numpy() == window

        for state in STATES:
            mask = window_mask & (states == state)

            ids = event_ids[mask]

            if len(ids) == 0:
                continue

            for side in SIDES:
                if side == "LONG":
                    favorable = cache["long_favorable"]

                    adverse = cache["long_adverse"]

                else:
                    favorable = cache["short_favorable"]

                    adverse = cache["short_adverse"]

                h = 30

                mfe = np.max(
                    favorable[
                        ids,
                        :h,
                    ],
                    axis=1,
                )

                mae = np.max(
                    adverse[
                        ids,
                        :h,
                    ],
                    axis=1,
                )

                rows.append(
                    {
                        "window": window,
                        "hmm_state": state,
                        "side": side,
                        "observations": len(ids),
                        "mean_mfe_30": safe_mean(mfe),
                        "median_mfe_30": safe_median(mfe),
                        "mean_mae_30": safe_mean(mae),
                        "median_mae_30": safe_median(mae),
                        "prob_mfe_ge_20": safe_ratio(
                            np.sum(mfe >= 20),
                            len(mfe),
                        ),
                        "prob_mae_ge_20": safe_ratio(
                            np.sum(mae >= 20),
                            len(mae),
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# STATE DISTRIBUTION BY WINDOW
# =============================================================================


def run_distribution_analysis(
    events: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = (
        events.groupby(
            [
                "window",
                "hmm_state",
            ]
        )
        .size()
        .reset_index(name="observations")
    )

    totals = events.groupby("window").size().rename("window_total").reset_index()

    result = grouped.merge(
        totals,
        on="window",
        how="left",
    )

    result["share"] = result["observations"] / result["window_total"]

    return result


# =============================================================================
# PRINT IMPORTANT RESULTS
# =============================================================================


def print_key_results(
    state_summary: pd.DataFrame,
    state_z: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:

    section("KEY DESCRIPTIVE RESULTS")

    print("STATE DISTRIBUTION")

    print(state_summary.to_string(index=False))

    print()

    if not transitions.empty:
        print("TRANSITION MATRIX")

        transition_matrix = transitions.pivot(
            index="from_state",
            columns="to_state",
            values="probability",
        )

        print(transition_matrix.to_string())

    print()

    if not state_z.empty:
        print("STATE × Z — 30 BAR SUMMARY")

        columns = [
            "side",
            "hmm_state",
            "zscore_threshold",
            "observations",
            "mean_mfe",
            "mean_mae",
            "mfe_ge_20",
            "mfe_ge_50",
            "mfe_ge_100",
        ]

        available = [c for c in columns if c in state_z.columns]

        # Keep only horizon 30.

        display = state_z.loc[
            state_z["horizon_bars"] == 30,
            available,
        ]

        print(display.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08C")

    print("HMM STATE INFORMATION ANALYSIS")

    print("-" * 100)

    print("No HMM retraining.")

    print("No SL/TP optimization.")

    print("No final strategy.")

    print("No volatility-regime interpretation.")

    print("Uses the clean causal HMM cache from Research 08B.")

    # =========================================================================
    # LOAD
    # =========================================================================

    metadata = load_metadata()

    states = load_hmm_states(metadata)

    cache = load_path_cache()

    # =========================================================================
    # BUILD EVENTS
    # =========================================================================

    events = build_events(
        metadata,
        states,
    )

    # =========================================================================
    # ANALYSES
    # =========================================================================

    state_summary = run_state_summary(events)

    state_z = run_state_z_analysis(
        events,
        cache,
    )

    state_path = run_state_path_summary(
        events,
        cache,
    )

    transitions = run_transition_analysis(events)

    windows = run_window_analysis(
        events,
        cache,
    )

    distribution = run_distribution_analysis(events)

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_summary.to_csv(
        OUTPUT_STATE,
        index=False,
    )

    state_z.to_csv(
        OUTPUT_STATE_Z,
        index=False,
    )

    state_path.to_csv(
        RESULTS_DIR / "research_08c_state_path_summary.csv",
        index=False,
    )

    transitions.to_csv(
        OUTPUT_TRANSITIONS,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    distribution.to_csv(
        OUTPUT_DISTRIBUTION,
        index=False,
    )

    # =========================================================================
    # PRINT
    # =========================================================================

    print_key_results(
        state_summary,
        state_z,
        transitions,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    section("RESEARCH 08C COMPLETE")

    print(f"Events analyzed: {len(events):,}")

    print()
    print("FILES SAVED")

    print(OUTPUT_STATE)

    print(OUTPUT_STATE_Z)

    print(RESULTS_DIR / "research_08c_state_path_summary.csv")

    print(OUTPUT_TRANSITIONS)

    print(OUTPUT_WINDOWS)

    print(OUTPUT_DISTRIBUTION)

    print()
    print("IMPORTANT:")

    print("HMM states remain raw labels 0 / 1 / 2.")

    print("No state was assumed to mean low, medium, or high volatility.")

    print("No parameter was optimized.")

    print("No strategy was constructed.")


if __name__ == "__main__":
    main()
