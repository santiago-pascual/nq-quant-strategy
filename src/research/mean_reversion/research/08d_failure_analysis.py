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

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT = RESULTS_DIR / "research_08d_failure_analysis.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08d_failure_by_window.csv"


# =============================================================================
# PARAMETERS
# =============================================================================

STATES = (0, 1, 2)

Z_THRESHOLDS = (
    1.5,
    2.0,
    2.5,
    3.0,
)

HORIZONS = (
    10,
    20,
    30,
    60,
    120,
)

TARGETS = (
    20.0,
    35.0,
    50.0,
    75.0,
    100.0,
)

ADVERSE_LEVELS = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
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
# SAFE STATISTICS
# =============================================================================


def finite(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return values[np.isfinite(values)]


def mean(values):

    values = finite(values)

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def median(values):

    values = finite(values)

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def percentile(
    values,
    q,
):

    values = finite(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            q,
        )
    )


def ratio(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


# =============================================================================
# LOAD METADATA
# =============================================================================


def load_metadata():

    section("LOADING RESEARCH 07 METADATA")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{METADATA_PATH}")

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
        raise RuntimeError("Missing columns: " + ", ".join(missing))

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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    df = df.sort_values("event_id").reset_index(drop=True)

    print(f"Events: {len(df):,}")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm():

    section("LOADING CAUSAL HMM STATES")

    if not HMM_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{HMM_PATH}\n\nRun Research 08B first.")

    df = pd.read_csv(HMM_PATH)

    required = [
        "event_id",
        "window",
        "timestamp",
        "hmm_state",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing HMM columns: " + ", ".join(missing))

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(
        df["hmm_state"],
        errors="raise",
    ).astype(int)

    return df


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{PATH_CACHE_PATH}")

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
            raise RuntimeError(f"Missing path array: {key}")

    paths = {key: np.asarray(archive[key]) for key in required}

    n = len(paths["future_close"])

    print(f"Events: {n:,}")

    print(f"Maximum horizon: {paths['future_close'].shape[1]}")

    return paths


# =============================================================================
# BUILD EVENT TABLE
# =============================================================================


def build_events(
    metadata,
    hmm,
):

    section("BUILDING EVENT TABLE")

    events = metadata.merge(
        hmm[
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
        raise RuntimeError("Event/HMM merge lost rows.")

    if events["hmm_state"].isna().any():
        raise RuntimeError("Missing HMM states.")

    print(f"Events: {len(events):,}")

    return events


# =============================================================================
# FIRST-PASSAGE ANALYSIS
# =============================================================================


def first_passage(
    favorable,
    adverse,
    target,
    adverse_level,
    horizon,
):

    h = min(
        horizon,
        favorable.shape[1],
    )

    f = favorable[
        :,
        :h,
    ]

    a = adverse[
        :,
        :h,
    ]

    target_hit = f >= target

    adverse_hit = a >= adverse_level

    target_any = target_hit.any(axis=1)

    adverse_any = adverse_hit.any(axis=1)

    # -------------------------------------------------------------------------
    # Find first target / adverse timestamps.
    # -------------------------------------------------------------------------

    target_time = np.where(
        target_any,
        np.argmax(
            target_hit,
            axis=1,
        ),
        h + 1,
    )

    adverse_time = np.where(
        adverse_any,
        np.argmax(
            adverse_hit,
            axis=1,
        ),
        h + 1,
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We classify according to which barrier was reached FIRST.
    #
    # If target and adverse barrier are reached on the same bar,
    # classify as FAILURE.
    # -------------------------------------------------------------------------

    success = target_time < adverse_time

    failure = adverse_time <= target_time

    never = ~target_any & ~adverse_any

    return (
        success,
        failure,
        never,
        target_time,
        adverse_time,
    )


# =============================================================================
# FAILURE ANALYSIS
# =============================================================================


def run_failure_analysis(
    events,
    paths,
):

    section("RUNNING FAILURE ANALYSIS")

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    states = events["hmm_state"].to_numpy(dtype=int)

    windows = events["window"].to_numpy(dtype=int)

    zscores = events["zscore_30"].to_numpy(dtype=float)

    rows = []

    total_tests = (
        2
        * len(STATES)
        * len(Z_THRESHOLDS)
        * len(TARGETS)
        * len(ADVERSE_LEVELS)
        * len(HORIZONS)
    )

    test_number = 0

    for side in ("LONG", "SHORT"):
        if side == "LONG":
            favorable = paths["long_favorable"]

            adverse = paths["long_adverse"]

            z_direction = -1

        else:
            favorable = paths["short_favorable"]

            adverse = paths["short_adverse"]

            z_direction = 1

        for state in STATES:
            state_mask = states == state

            for z in Z_THRESHOLDS:
                if z_direction < 0:
                    z_mask = zscores <= -z

                else:
                    z_mask = zscores >= z

                base_mask = state_mask & z_mask

                ids = event_ids[base_mask]

                if len(ids) == 0:
                    continue

                local_favorable = favorable[ids]

                local_adverse = adverse[ids]

                local_windows = windows[base_mask]

                for target in TARGETS:
                    for adverse_level in ADVERSE_LEVELS:
                        for horizon in HORIZONS:
                            test_number += 1

                            if test_number % 250 == 0:
                                print(f"  test {test_number:,}/{total_tests:,}")

                            (
                                success,
                                failure,
                                never,
                                target_time,
                                adverse_time,
                            ) = first_passage(
                                local_favorable,
                                local_adverse,
                                target,
                                adverse_level,
                                horizon,
                            )

                            n = len(ids)

                            successful_ids = ids[success]

                            failed_ids = ids[failure]

                            # -------------------------------------------------
                            # For successful trades:
                            # how much adverse movement occurred BEFORE target?
                            # -------------------------------------------------

                            successful_adv = local_adverse[success]

                            successful_target_times = target_time[success]

                            if len(successful_adv) > 0:
                                pre_target_mae = []

                                for i, t in enumerate(successful_target_times):
                                    pre_target_mae.append(
                                        np.max(
                                            successful_adv[
                                                i,
                                                : t + 1,
                                            ]
                                        )
                                    )

                                pre_target_mae = np.asarray(pre_target_mae)

                            else:
                                pre_target_mae = np.array(
                                    [],
                                    dtype=float,
                                )

                            # -------------------------------------------------
                            # For failures:
                            # how far did price travel favorably before
                            # the adverse barrier?
                            # -------------------------------------------------

                            failed_fav = local_favorable[failure]

                            failed_adverse_times = adverse_time[failure]

                            if len(failed_fav) > 0:
                                pre_failure_mfe = []

                                for i, t in enumerate(failed_adverse_times):
                                    pre_failure_mfe.append(
                                        np.max(
                                            failed_fav[
                                                i,
                                                : t + 1,
                                            ]
                                        )
                                    )

                                pre_failure_mfe = np.asarray(pre_failure_mfe)

                            else:
                                pre_failure_mfe = np.array(
                                    [],
                                    dtype=float,
                                )

                            rows.append(
                                {
                                    "side": side,
                                    "hmm_state": state,
                                    "zscore_threshold": z,
                                    "target": target,
                                    "adverse_level": adverse_level,
                                    "horizon": horizon,
                                    "observations": n,
                                    "success_count": int(success.sum()),
                                    "failure_count": int(failure.sum()),
                                    "never_count": int(never.sum()),
                                    "success_rate": ratio(
                                        success.sum(),
                                        n,
                                    ),
                                    "failure_rate": ratio(
                                        failure.sum(),
                                        n,
                                    ),
                                    "never_rate": ratio(
                                        never.sum(),
                                        n,
                                    ),
                                    "median_time_to_target": median(
                                        target_time[success]
                                    ),
                                    "mean_time_to_target": mean(target_time[success]),
                                    "median_time_to_failure": median(
                                        adverse_time[failure]
                                    ),
                                    "mean_time_to_failure": mean(adverse_time[failure]),
                                    "successful_pre_target_mae_mean": mean(
                                        pre_target_mae
                                    ),
                                    "successful_pre_target_mae_median": median(
                                        pre_target_mae
                                    ),
                                    "successful_pre_target_mae_p75": percentile(
                                        pre_target_mae,
                                        75,
                                    ),
                                    "successful_pre_target_mae_p90": percentile(
                                        pre_target_mae,
                                        90,
                                    ),
                                    "failed_pre_failure_mfe_mean": mean(
                                        pre_failure_mfe
                                    ),
                                    "failed_pre_failure_mfe_median": median(
                                        pre_failure_mfe
                                    ),
                                    "failed_pre_failure_mfe_p75": percentile(
                                        pre_failure_mfe,
                                        75,
                                    ),
                                    "failed_pre_failure_mfe_p90": percentile(
                                        pre_failure_mfe,
                                        90,
                                    ),
                                }
                            )

    return pd.DataFrame(rows)


# =============================================================================
# WINDOW STABILITY
# =============================================================================


def run_window_analysis(
    events,
    paths,
):

    section("WINDOW-BY-WINDOW FAILURE STABILITY")

    rows = []

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    states = events["hmm_state"].to_numpy(dtype=int)

    windows = events["window"].to_numpy(dtype=int)

    zscores = events["zscore_30"].to_numpy(dtype=float)

    for side in ("LONG", "SHORT"):
        if side == "LONG":
            favorable = paths["long_favorable"]

            adverse = paths["long_adverse"]

        else:
            favorable = paths["short_favorable"]

            adverse = paths["short_adverse"]

        for state in STATES:
            for z in Z_THRESHOLDS:
                if side == "LONG":
                    base = (states == state) & (zscores <= -z)

                else:
                    base = (states == state) & (zscores >= z)

                for window in np.unique(windows):
                    mask = base & (windows == window)

                    ids = event_ids[mask]

                    if len(ids) == 0:
                        continue

                    local_f = favorable[
                        ids,
                        :30,
                    ]

                    local_a = adverse[
                        ids,
                        :30,
                    ]

                    mfe = np.max(
                        local_f,
                        axis=1,
                    )

                    mae = np.max(
                        local_a,
                        axis=1,
                    )

                    rows.append(
                        {
                            "side": side,
                            "window": window,
                            "hmm_state": state,
                            "zscore_threshold": z,
                            "observations": len(ids),
                            "mean_mfe_30": mean(mfe),
                            "median_mfe_30": median(mfe),
                            "mean_mae_30": mean(mae),
                            "median_mae_30": median(mae),
                            "mfe_ge_20": ratio(
                                np.sum(mfe >= 20),
                                len(mfe),
                            ),
                            "mfe_ge_50": ratio(
                                np.sum(mfe >= 50),
                                len(mfe),
                            ),
                            "mfe_ge_100": ratio(
                                np.sum(mfe >= 100),
                                len(mfe),
                            ),
                            "mae_ge_10": ratio(
                                np.sum(mae >= 10),
                                len(mae),
                            ),
                            "mae_ge_20": ratio(
                                np.sum(mae >= 20),
                                len(mae),
                            ),
                            "mae_ge_35": ratio(
                                np.sum(mae >= 35),
                                len(mae),
                            ),
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# FAILURE PROFILE
# =============================================================================


def print_failure_profile(
    df,
):

    section("FAILURE PROFILE")

    if df.empty:
        print("No results.")

        return

    # -------------------------------------------------------------------------
    # Focus on the combinations that previously looked strongest.
    # -------------------------------------------------------------------------

    candidates = [
        (
            "LONG",
            1,
            2.0,
        ),
        (
            "LONG",
            1,
            2.5,
        ),
        (
            "LONG",
            1,
            3.0,
        ),
        (
            "LONG",
            2,
            3.0,
        ),
        (
            "SHORT",
            1,
            2.0,
        ),
        (
            "SHORT",
            1,
            2.5,
        ),
        (
            "SHORT",
            1,
            3.0,
        ),
    ]

    for side, state, z in candidates:
        subset = df[
            (df["side"] == side)
            & (df["hmm_state"] == state)
            & (df["zscore_threshold"] == z)
            & (df["horizon"] == 60)
        ]

        if subset.empty:
            continue

        # Pick TP=50 for the diagnostic table.
        subset = subset[subset["target"] == 50]

        if subset.empty:
            continue

        print()
        print(f"{side} | STATE={state} | Z={z} | TP=50 | H=60")

        columns = [
            "adverse_level",
            "observations",
            "success_rate",
            "failure_rate",
            "never_rate",
            "median_time_to_target",
            "median_time_to_failure",
            "successful_pre_target_mae_median",
            "successful_pre_target_mae_p75",
            "successful_pre_target_mae_p90",
            "failed_pre_failure_mfe_median",
        ]

        print(subset[columns].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08D")

    print("FAILURE MODE ANALYSIS")

    print("-" * 100)

    print("No HMM retraining.")

    print("No parameter optimization.")

    print("No final strategy.")

    print("No volatility-regime filtering.")

    print("Purpose: understand why trades fail before large favorable moves.")

    # =========================================================================
    # LOAD
    # =========================================================================

    metadata = load_metadata()

    hmm = load_hmm()

    paths = load_paths()

    events = build_events(
        metadata,
        hmm,
    )

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    results = run_failure_analysis(
        events,
        paths,
    )

    window_results = run_window_analysis(
        events,
        paths,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    window_results.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    # =========================================================================
    # PRINT
    # =========================================================================

    print_failure_profile(results)

    # =========================================================================
    # FINAL
    # =========================================================================

    section("RESEARCH 08D COMPLETE")

    print(f"Failure-analysis rows: {len(results):,}")

    print(f"Window-analysis rows: {len(window_results):,}")

    print()
    print("Saved:")

    print(OUTPUT)

    print(OUTPUT_WINDOWS)

    print()
    print("Interpretation:")

    print("SUCCESS = target reached before adverse barrier.")

    print("FAILURE = adverse barrier reached before or on the same bar as target.")

    print("NEVER = neither barrier reached inside the horizon.")

    print()
    print("No SL/TP was selected as a strategy.")


if __name__ == "__main__":
    main()
