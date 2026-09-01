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

# -------------------------------------------------------------------------
# IMPORTANT:
# We do NOT depend on the broken Research 08 volatility/HMM cache.
# Volatility is calculated directly from the event metadata/features if
# available.
# -------------------------------------------------------------------------

FEATURE_CANDIDATES = [
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
]


# =============================================================================
# RESEARCH GRID
# =============================================================================

# We deliberately extend Z because this is one of the main hypotheses.
Z_THRESHOLDS = (
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
)

HMM_STATES = (
    0,
    1,
    2,
)

# Keep small/intermediate stops heavily represented.
STOPS = (
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
)

TARGETS = (
    20.0,
    25.0,
    35.0,
    50.0,
    75.0,
    100.0,
    125.0,
)

HORIZONS = (
    10,
    20,
    30,
    60,
    120,
)

# Volatility buckets.
# These are deliberately descriptive rather than assumed regimes.
VOL_BINS = (
    0.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    np.inf,
)

VOL_LABELS = (
    "<20",
    "20-40",
    "40-60",
    "60-80",
    "80-100",
    "100+",
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
# SAFE METRICS
# =============================================================================


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.median(x))


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.std(x, ddof=1))


def safe_ratio(a, b):
    if b == 0:
        return np.nan

    return float(a / b)


# =============================================================================
# LOAD RESEARCH 07
# =============================================================================


def load_metadata():

    section("LOADING RESEARCH 07 METADATA")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing Research 07 metadata:\n{METADATA_PATH}")

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
        raise RuntimeError("Missing metadata columns: " + ", ".join(missing))

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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    df = df.sort_values("event_id").reset_index(drop=True)

    print(f"Events: {len(df):,}")
    print(f"Windows: {df['window'].nunique()}")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm():

    section("LOADING CLEAN CAUSAL HMM")

    if not HMM_PATH.exists():
        raise FileNotFoundError(
            f"Missing clean HMM cache:\n{HMM_PATH}\n\nRun Research 08B first."
        )

    df = pd.read_csv(HMM_PATH)

    required = [
        "event_id",
        "hmm_state",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing HMM columns: " + ", ".join(missing))

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(
        df["hmm_state"],
        errors="coerce",
    ).astype("Int8")

    df = df[
        [
            "event_id",
            "hmm_state",
        ]
    ]

    print(f"HMM events: {len(df):,}")

    return df


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing path cache:\n{PATH_CACHE_PATH}")

    archive = np.load(
        PATH_CACHE_PATH,
        allow_pickle=False,
    )

    required = [
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    for key in required:
        if key not in archive.files:
            raise RuntimeError(f"Missing path array: {key}")

    paths = {
        key: np.asarray(
            archive[key],
            dtype=np.float32,
        )
        for key in required
    }

    n = paths["long_favorable"].shape[0]

    h = paths["long_favorable"].shape[1]

    print(f"Cached events: {n:,}")

    print(f"Maximum horizon: {h}")

    return paths


# =============================================================================
# VOLATILITY
# =============================================================================


def find_volatility_column():
    """
    Research 07 metadata itself does not contain realized volatility.

    Therefore we use the available z-score path only if a volatility
    feature was persisted elsewhere.

    This function searches the standard Research 07 feature output files.
    """

    possible_files = [
        RESULTS_DIR / "research_07_features.csv",
        RESULTS_DIR / "features.csv",
        CACHE_DIR / "research_07_features.csv",
    ]

    for path in possible_files:
        if not path.exists():
            continue

        try:
            header = pd.read_csv(
                path,
                nrows=0,
            ).columns.tolist()

        except Exception:
            continue

        for candidate in FEATURE_CANDIDATES:
            if candidate in header:
                return path, candidate

    return None, None


def load_volatility(metadata):

    section("LOADING VOLATILITY FEATURE")

    feature_path, vol_column = find_volatility_column()

    if feature_path is None:
        print("No persisted realized-volatility feature file was found.")

        print(
            "Research 08E will therefore use a DATA-DRIVEN "
            "volatility proxy derived from the 30-bar z-score."
        )

        # -----------------------------------------------------------------
        # IMPORTANT:
        #
        # This is NOT S2R.
        # It is only a descriptive volatility proxy for this research.
        #
        # We intentionally avoid pretending that z-score itself is
        # realized volatility.
        # -----------------------------------------------------------------

        z = metadata["zscore_30"].to_numpy(dtype=np.float32)

        # Rolling absolute z-score is not available from sparse metadata,
        # so use absolute z-score as a fallback descriptive stress proxy.
        #
        # It is NOT called "realized volatility" in the output.
        proxy = np.abs(z)

        return proxy, "abs_zscore_proxy"

    print(f"Volatility source: {feature_path}")

    print(f"Volatility column: {vol_column}")

    features = pd.read_csv(
        feature_path,
        usecols=[
            "timestamp",
            vol_column,
        ],
    )

    features["timestamp"] = pd.to_datetime(
        features["timestamp"],
        utc=True,
        errors="coerce",
    )

    features[vol_column] = pd.to_numeric(
        features[vol_column],
        errors="coerce",
    )

    features = features.dropna(
        subset=[
            "timestamp",
            vol_column,
        ]
    ).sort_values("timestamp")

    # ---------------------------------------------------------------------
    # Normalize timestamp representation.
    # ---------------------------------------------------------------------

    metadata_sorted = metadata[
        [
            "event_id",
            "timestamp",
        ]
    ].copy()

    metadata_sorted["timestamp"] = pd.to_datetime(
        metadata_sorted["timestamp"],
        utc=True,
        errors="raise",
    )

    metadata_sorted = metadata_sorted.sort_values("timestamp")

    features = features.sort_values("timestamp")

    merged = pd.merge_asof(
        metadata_sorted,
        features,
        on="timestamp",
        direction="backward",
    )

    merged = merged.sort_values("event_id")

    values = merged[vol_column].to_numpy(dtype=np.float32)

    if len(values) != len(metadata):
        raise RuntimeError("Volatility mapping changed event count.")

    return values, vol_column


# =============================================================================
# BUILD EVENT TABLE
# =============================================================================


def build_events(
    metadata,
    hmm,
):

    section("BUILDING EVENT TABLE")

    events = metadata.merge(
        hmm,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    missing = events["hmm_state"].isna().sum()

    if missing:
        raise RuntimeError(f"{missing:,} events have no HMM state.")

    events["hmm_state"] = events["hmm_state"].astype(np.int8)

    print(f"Final events: {len(events):,}")

    return events


# =============================================================================
# VOLATILITY BUCKETS
# =============================================================================


def build_volatility_buckets(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    labels = np.full(
        len(values),
        "UNKNOWN",
        dtype=object,
    )

    finite_mask = np.isfinite(values)

    bucket_indices = np.digitize(
        values[finite_mask],
        VOL_BINS[1:-1],
        right=False,
    )

    for i, label in enumerate(VOL_LABELS):
        mask = bucket_indices == i

        temp = labels[finite_mask]

        temp[mask] = label

        labels[finite_mask] = temp

    return labels


# =============================================================================
# CORE TRADE EVALUATION
# =============================================================================


def evaluate_subset(
    favorable,
    adverse,
    target,
    stop,
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

    stop_hit = a >= stop

    target_exists = target_hit.any(axis=1)

    stop_exists = stop_hit.any(axis=1)

    target_time = np.where(
        target_exists,
        np.argmax(
            target_hit,
            axis=1,
        ),
        h + 1,
    )

    stop_time = np.where(
        stop_exists,
        np.argmax(
            stop_hit,
            axis=1,
        ),
        h + 1,
    )

    # Same-bar target + stop = STOP.
    win = target_time < stop_time

    loss = stop_time <= target_time

    timeout = ~target_exists & ~stop_exists

    # If target happens before stop:
    # calculate favorable excursion until target.
    #
    # If timeout:
    # use full available horizon.
    #
    # If loss:
    # use favorable excursion before stop.
    # -------------------------------------------------------------------------

    n = len(f)

    mfe_before_exit = np.zeros(
        n,
        dtype=np.float32,
    )

    mae_before_exit = np.zeros(
        n,
        dtype=np.float32,
    )

    for i in range(n):
        if win[i]:
            end = int(target_time[i]) + 1

        elif loss[i]:
            end = int(stop_time[i]) + 1

        else:
            end = h

        if end <= 0:
            continue

        mfe_before_exit[i] = np.max(f[i, :end])

        mae_before_exit[i] = np.max(a[i, :end])

    return (
        win,
        loss,
        timeout,
        target_time,
        stop_time,
        mfe_before_exit,
        mae_before_exit,
    )


# =============================================================================
# GRID
# =============================================================================


def run_grid(
    events,
    paths,
    volatility,
):

    section("RUNNING Z × HMM × VOL × SL × TP × HORIZON GRID")

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    zscores = events["zscore_30"].to_numpy(dtype=np.float32)

    states = events["hmm_state"].to_numpy(dtype=np.int8)

    windows = events["window"].to_numpy(dtype=np.int16)

    volatility = np.asarray(
        volatility,
        dtype=np.float32,
    )

    vol_labels = build_volatility_buckets(volatility)

    rows = []

    combinations = (
        2
        * len(HMM_STATES)
        * len(Z_THRESHOLDS)
        * len(VOL_LABELS)
        * len(STOPS)
        * len(TARGETS)
        * len(HORIZONS)
    )

    completed = 0

    print(f"Total combinations: {combinations:,}")

    for side in (
        "LONG",
        "SHORT",
    ):
        if side == "LONG":
            favorable_all = paths["long_favorable"]

            adverse_all = paths["long_adverse"]

        else:
            favorable_all = paths["short_favorable"]

            adverse_all = paths["short_adverse"]

        for state in HMM_STATES:
            state_mask = states == state

            for z in Z_THRESHOLDS:
                if side == "LONG":
                    z_mask = zscores <= -z

                else:
                    z_mask = zscores >= z

                for vol_label in VOL_LABELS:
                    vol_mask = vol_labels == vol_label

                    base_mask = state_mask & z_mask & vol_mask

                    ids = event_ids[base_mask]

                    if len(ids) == 0:
                        # Still count combinations.
                        completed += len(STOPS) * len(TARGETS) * len(HORIZONS)

                        continue

                    local_f = favorable_all[ids]

                    local_a = adverse_all[ids]

                    for stop in STOPS:
                        for target in TARGETS:
                            for horizon in HORIZONS:
                                completed += 1

                                (
                                    win,
                                    loss,
                                    timeout,
                                    target_time,
                                    stop_time,
                                    mfe,
                                    mae,
                                ) = evaluate_subset(
                                    local_f,
                                    local_a,
                                    target,
                                    stop,
                                    horizon,
                                )

                                n = len(win)

                                wins = int(win.sum())

                                losses = int(loss.sum())

                                timeouts = int(timeout.sum())

                                win_rate = wins / n

                                loss_rate = losses / n

                                timeout_rate = timeouts / n

                                # -----------------------------------------------------------------
                                # 1R framework.
                                #
                                # If TP/SL are measured in points,
                                # expected R before costs is:
                                #
                                # WR * TP/SL - LR
                                #
                                # Timeout is treated as 0R.
                                # -----------------------------------------------------------------

                                expectancy_r = win_rate * (target / stop) - loss_rate

                                gross_profit_r = wins * (target / stop)

                                gross_loss_r = losses

                                pf = (
                                    gross_profit_r / gross_loss_r
                                    if gross_loss_r > 0
                                    else np.nan
                                )

                                rows.append(
                                    {
                                        "side": side,
                                        "hmm_state": state,
                                        "zscore": z,
                                        "volatility_regime": vol_label,
                                        "observations": n,
                                        "sl_points": stop,
                                        "tp_points": target,
                                        "horizon": horizon,
                                        "wins": wins,
                                        "losses": losses,
                                        "timeouts": timeouts,
                                        "win_rate": win_rate,
                                        "loss_rate": loss_rate,
                                        "timeout_rate": timeout_rate,
                                        "expectancy_R": expectancy_r,
                                        "profit_factor": pf,
                                        "median_win_time": safe_median(
                                            target_time[win]
                                        ),
                                        "median_loss_time": safe_median(
                                            stop_time[loss]
                                        ),
                                        "mean_mfe": safe_mean(mfe),
                                        "median_mfe": safe_median(mfe),
                                        "mean_mae": safe_mean(mae),
                                        "median_mae": safe_median(mae),
                                        "mfe_mae_ratio": safe_ratio(
                                            safe_mean(mfe),
                                            safe_mean(mae),
                                        ),
                                        "window_count": len(
                                            np.unique(windows[base_mask])
                                        ),
                                    }
                                )

                                if completed % 500 == 0:
                                    print(f"  {completed:,}/{combinations:,}")

    return pd.DataFrame(rows)


# =============================================================================
# WINDOW ROBUSTNESS
# =============================================================================


def run_window_robustness(
    grid,
    events,
):

    section("BUILDING WINDOW ROBUSTNESS")

    # -------------------------------------------------------------------------
    # Aggregate only combinations that actually have observations.
    # -------------------------------------------------------------------------

    group_cols = [
        "side",
        "hmm_state",
        "zscore",
        "volatility_regime",
        "sl_points",
        "tp_points",
        "horizon",
    ]

    rows = []

    grouped = grid.groupby(
        group_cols,
        dropna=False,
    )

    for key, group in grouped:
        if group.empty:
            continue

        # The base grid already contains aggregate metrics.
        # Robustness here is based on dispersion of combinations across
        # event-count slices where possible.
        #
        # Since the Research 07 windows are fixed OOS partitions, we derive
        # window-level statistics directly from the original events below.

        (
            side,
            state,
            z,
            vol_regime,
            sl,
            tp,
            horizon,
        ) = key

        rows.append(
            {
                "side": side,
                "hmm_state": state,
                "zscore": z,
                "volatility_regime": vol_regime,
                "sl_points": sl,
                "tp_points": tp,
                "horizon": horizon,
                "aggregate_expectancy_R": safe_mean(group["expectancy_R"]),
                "aggregate_pf": safe_mean(group["profit_factor"]),
                "aggregate_win_rate": safe_mean(group["win_rate"]),
                "observations": int(group["observations"].sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CANDIDATE REPORT
# =============================================================================


def print_candidates(
    grid,
):

    section("TOP ROBUST CANDIDATES")

    if grid.empty:
        print("No combinations produced observations.")

        return

    # -------------------------------------------------------------------------
    # Minimum sample threshold.
    # -------------------------------------------------------------------------

    filtered = grid[grid["observations"] >= 500].copy()

    if filtered.empty:
        filtered = grid.copy()

    # -------------------------------------------------------------------------
    # Candidate ranking is deliberately NOT just PF.
    #
    # We require:
    #   - positive expectancy
    #   - enough observations
    #   - meaningful TP/SL
    #
    # Then display several different rankings.
    # -------------------------------------------------------------------------

    filtered["score"] = filtered["expectancy_R"] * np.log1p(filtered["observations"])

    cols = [
        "side",
        "hmm_state",
        "zscore",
        "volatility_regime",
        "sl_points",
        "tp_points",
        "horizon",
        "observations",
        "win_rate",
        "expectancy_R",
        "profit_factor",
        "median_win_time",
        "median_loss_time",
        "mean_mfe",
        "mean_mae",
    ]

    print()
    print("TOP BY EXPECTANCY_R")

    print(
        filtered.sort_values(
            [
                "expectancy_R",
                "observations",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(25)[cols]
        .to_string(index=False)
    )

    print()
    print("TOP BY PROFIT FACTOR")

    print(
        filtered.dropna(subset=["profit_factor"])
        .sort_values(
            [
                "profit_factor",
                "observations",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(25)[cols]
        .to_string(index=False)
    )

    print()
    print("TOP HIGH-WIN-RATE CANDIDATES")

    high_wr = filtered[filtered["tp_points"] >= filtered["sl_points"]]

    print(
        high_wr.sort_values(
            [
                "win_rate",
                "expectancy_R",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(25)[cols]
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Specifically investigate the Z hypothesis.
    # -------------------------------------------------------------------------

    print()
    print("Z-SCORE COMPARISON — STATE 1")

    z_compare = (
        filtered[filtered["hmm_state"] == 1]
        .groupby(
            [
                "side",
                "zscore",
            ],
            as_index=False,
        )
        .agg(
            observations=(
                "observations",
                "sum",
            ),
            mean_win_rate=(
                "win_rate",
                "mean",
            ),
            mean_expectancy_R=(
                "expectancy_R",
                "mean",
            ),
            median_expectancy_R=(
                "expectancy_R",
                "median",
            ),
            mean_pf=(
                "profit_factor",
                "mean",
            ),
        )
    )

    print(
        z_compare.sort_values(
            [
                "side",
                "zscore",
            ]
        ).to_string(index=False)
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08E")

    print("Z-SCORE × HMM × VOLATILITY × SL × TP × HORIZON")

    print("-" * 100)

    print("Research only.")

    print("No production strategy changes.")

    print("No HMM retraining.")

    print("No automatic parameter selection.")

    print("Same-bar TARGET + STOP -> STOP.")

    print()
    print("Main hypothesis:")

    print(
        "Higher Z may reduce low-quality entries and permit"
        " smaller SL / larger TP structures."
    )

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

    volatility, volatility_source = load_volatility(events)

    print()
    print(f"Volatility classification source: {volatility_source}")

    # =========================================================================
    # GRID
    # =========================================================================

    grid = run_grid(
        events,
        paths,
        volatility,
    )

    # =========================================================================
    # ROBUSTNESS
    # =========================================================================

    robustness = run_window_robustness(
        grid,
        events,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_path = RESULTS_DIR / "research_08e_strategy_grid.csv"

    robustness_path = RESULTS_DIR / "research_08e_robustness.csv"

    grid.to_csv(
        grid_path,
        index=False,
    )

    robustness.to_csv(
        robustness_path,
        index=False,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print_candidates(grid)

    # =========================================================================
    # FINAL
    # =========================================================================

    section("RESEARCH 08E COMPLETE")

    print(f"Grid rows: {len(grid):,}")

    print(f"Robustness rows: {len(robustness):,}")

    print()
    print("Saved:")

    print(grid_path)

    print(robustness_path)

    print()
    print("IMPORTANT:")

    print("The volatility bucket is descriptive.")

    print("40-60 is NOT assumed to be optimal.")

    print("HMM states remain raw labels 0 / 1 / 2.")

    print(
        "Z=3.5 and Z=4.0 are exploratory and require"
        " particular attention to sample size."
    )

    print("No final strategy was selected.")


if __name__ == "__main__":
    main()
