from __future__ import annotations

from pathlib import Path
from bisect import bisect_right, insort
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# RESEARCH 08U
# TRUE PARAMETER NEIGHBORHOOD ROBUSTNESS
# =============================================================================
#
# PURPOSE
# -------
# Determine whether the frozen MRS2 / MRL1 / MRL2 parameter points are located
# inside a locally robust region.
#
# THIS IS NOT OPTIMIZATION.
#
# Frozen candidates:
#
# MRS2 = SHORT | HMM 2 | VOL 80-100 | Z 2.0 | TP 5 | SL 2 | H 5
# MRL1 = LONG  | HMM 1 | VOL 20-40  | Z 2.5 | TP 5 | SL 2 | H 20
# MRL2 = LONG  | HMM 2 | VOL 60-80  | Z 3.5 | TP 5 | SL 2 | H 2
#
# Local neighborhood:
#
# TP = frozen - 0.50 / frozen / frozen + 0.50
# SL = frozen - 0.50 / frozen / frozen + 0.50
# H  = frozen - 1    / frozen / frozen + 1
#
# Three 2D surfaces are evaluated:
#
#   TP x SL @ frozen H
#   TP x H  @ frozen SL
#   SL x H  @ frozen TP
#
# The frozen candidate is NEVER replaced by the best neighbor.
#
# Path resolution:
# ----------------
# Research 07 future path cache is used directly.
#
# A trade is:
#
#   WIN       if TP is reached first
#   LOSS      if SL is reached first
#   AMBIGUOUS if TP and SL are first reached on the same bar
#   UNRESOLVED if neither threshold is reached within H
#
# Ambiguous and unresolved trades are excluded from resolved WR/PF.
# Their R contribution is zero in expectancy_all.
#
# No HMM retraining.
# No volatility optimization.
# No candidate replacement.
# No production changes.
# =============================================================================


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

OUT_DIR = RESULTS_DIR / "research_08u_parameter_neighborhood"

TABLES_DIR = OUT_DIR / "tables"

PLOTS_DIR = OUT_DIR / "plots"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"


# =============================================================================
# FROZEN CANDIDATES
# =============================================================================

CANDIDATES = {
    "MRS2": {
        "candidate_id": "C01",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "horizon": 5,
    },
    "MRL1": {
        "candidate_id": "C02",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "horizon": 20,
    },
    "MRL2": {
        "candidate_id": "C06",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "horizon": 2,
    },
}


# =============================================================================
# PARAMETERS
# =============================================================================

TP_STEP = 0.50
SL_STEP = 0.50
H_STEP = 1

MIN_OBSERVATIONS = 20

EPS = 1e-12


# =============================================================================
# UTILITIES
# =============================================================================


def section(title: str):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def create_directories():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# RESEARCH 07
# =============================================================================


def load_research_07():

    section("LOADING RESEARCH 07")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing Research 07 metadata:\n{METADATA_PATH}")

    if not PATH_CACHE.exists():
        raise FileNotFoundError(f"Missing Research 07 path cache:\n{PATH_CACHE}")

    metadata = pd.read_csv(METADATA_PATH).reset_index(drop=True)

    print(f"Metadata rows: {len(metadata):,}")

    z = np.load(
        PATH_CACHE,
        allow_pickle=False,
    )

    required = [
        "future_close",
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    missing = [key for key in required if key not in z.files]

    if missing:
        raise RuntimeError(f"Research 07 path cache missing:\n{missing}")

    arrays = {key: z[key] for key in required}

    print()
    print("Cache keys:")

    for key, arr in arrays.items():
        print(f"{key}: shape={arr.shape} dtype={arr.dtype}")

    expected = len(metadata)

    for key, arr in arrays.items():
        if arr.ndim != 2:
            raise RuntimeError(f"{key} is not 2-dimensional.")

        if arr.shape[0] != expected:
            raise RuntimeError(
                f"Research 07 alignment failure:\n"
                f"{key}: {arr.shape[0]:,}\n"
                f"metadata: {expected:,}"
            )

    max_horizon = arrays["future_close"].shape[1]

    print()
    print("Research 07 cache integrity: OK")

    print(f"Events: {expected:,}")

    print(f"Maximum horizon: {max_horizon}")

    return metadata, arrays


# =============================================================================
# HMM
# =============================================================================


def load_hmm():

    section("LOADING RESEARCH 08B HMM")

    if not HMM_PATH.exists():
        raise FileNotFoundError(f"Missing Research 08B HMM:\n{HMM_PATH}")

    hmm = pd.read_csv(HMM_PATH).reset_index(drop=True)

    print(f"Using Research 08B HMM cache:\n{HMM_PATH}")

    state_column = None

    for column in [
        "hmm_state",
        "HMM_state",
        "hmm_states",
        "state",
    ]:
        if column in hmm.columns:
            state_column = column
            break

    if state_column is None:
        candidates = [
            c
            for c in hmm.columns
            if ("hmm" in str(c).lower() and "state" in str(c).lower())
        ]

        if len(candidates) == 1:
            state_column = candidates[0]

    if state_column is None:
        raise RuntimeError(
            f"Could not identify HMM state column.\nColumns: {list(hmm.columns)}"
        )

    hmm["hmm_state"] = pd.to_numeric(
        hmm[state_column],
        errors="coerce",
    )

    print(f"HMM rows: {len(hmm):,}")

    if hmm["hmm_state"].isna().any():
        raise RuntimeError("Missing HMM states detected.")

    print()
    print("HMM state distribution:")

    print(hmm["hmm_state"].astype(int).value_counts().sort_index())

    states = set(hmm["hmm_state"].astype(int).unique())

    if states != {0, 1, 2}:
        raise RuntimeError(f"Unexpected HMM states: {sorted(states)}")

    print()
    print("Research 08B HMM integrity: OK")

    return hmm[["hmm_state"]].reset_index(drop=True)


# =============================================================================
# MARKET DATA / VOLATILITY
# =============================================================================


def load_market():

    try:
        from src.databento_loader import load_databento_mnq

    except ImportError as exc:
        raise ImportError(
            "Could not import src.databento_loader.load_databento_mnq()."
        ) from exc

    print("Using project loader:")

    print("src.databento_loader.load_databento_mnq()")

    market = load_databento_mnq()

    print(f"Rows loaded: {len(market):,}")

    return market


def build_market_volatility():

    market = load_market()

    # -------------------------------------------------------------------------
    # EXACT REAL PROJECT TIMESTAMP
    # -------------------------------------------------------------------------

    if "timestamp ET" not in market.columns:
        raise RuntimeError(
            "Project market data does not contain "
            "'timestamp ET'.\n"
            f"Columns: {list(market.columns)}"
        )

    if "close" not in market.columns:
        raise RuntimeError("Project market data does not contain 'close'.")

    market = market.copy()

    market["_timestamp"] = pd.to_datetime(
        market["timestamp ET"],
        utc=True,
    )

    market = market.sort_values("_timestamp").reset_index(drop=True)

    print("Using market timestamp: timestamp ET")

    # -------------------------------------------------------------------------
    # Return / realized volatility
    # -------------------------------------------------------------------------

    print("Building return features...")

    close = pd.to_numeric(
        market["close"],
        errors="coerce",
    )

    returns = close.pct_change()

    print("Building volatility features...")

    realized_vol_30 = returns.rolling(
        window=30,
        min_periods=30,
    ).std()

    market["realized_vol_30"] = realized_vol_30

    valid = market[market["realized_vol_30"].notna()][
        [
            "_timestamp",
            "realized_vol_30",
        ]
    ].copy()

    print(f"Valid realized_vol_30: {len(valid):,}")

    if valid.empty:
        raise RuntimeError("No valid realized_vol_30.")

    return valid


# =============================================================================
# EVENT TIMESTAMP
# =============================================================================


def identify_event_timestamp_column(metadata):

    candidates = [
        "timestamp",
        "timestamp ET",
        "event_timestamp",
        "ts_event",
        "datetime",
    ]

    for column in candidates:
        if column in metadata.columns:
            return column

    raise RuntimeError(
        "Could not identify Research 07 event timestamp.\n"
        f"Columns: {list(metadata.columns)}"
    )


# =============================================================================
# CAUSAL VOLATILITY CONTEXT
# =============================================================================


def build_volatility_context(metadata):

    section("BUILDING CAUSAL VOLATILITY CONTEXT")

    market = build_market_volatility()

    event_timestamp_column = identify_event_timestamp_column(metadata)

    print(f"Using event timestamp column: {event_timestamp_column}")

    events = pd.DataFrame(
        {
            "_timestamp": pd.to_datetime(
                metadata[event_timestamp_column],
                utc=True,
            )
        }
    )

    # -------------------------------------------------------------------------
    # FORCE SAME pandas dtype.
    # -------------------------------------------------------------------------

    events["_timestamp"] = events["_timestamp"].astype("datetime64[ns, UTC]")

    market["_timestamp"] = market["_timestamp"].astype("datetime64[ns, UTC]")

    events = events.sort_values("_timestamp").reset_index(drop=True)

    market = market.sort_values("_timestamp").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Causal mapping.
    #
    # direction=backward means market timestamp <= event timestamp.
    # -------------------------------------------------------------------------

    print()
    print("Mapping events to realized_vol_30...")

    mapped = pd.merge_asof(
        events,
        market,
        on="_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    mapped_count = int(mapped["realized_vol_30"].notna().sum())

    print(f"Events mapped: {mapped_count:,}/{len(metadata):,}")

    if mapped_count == 0:
        raise RuntimeError("No Research 07 events mapped to realized_vol_30.")

    # -------------------------------------------------------------------------
    # Causal percentile.
    #
    # Each observation is ranked against the historical observations available
    # up to that point.
    # -------------------------------------------------------------------------

    print()
    print("Building causal volatility percentile...")

    values = mapped["realized_vol_30"].to_numpy(dtype=float)

    percentile = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    history = []

    total = len(values)

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        position = bisect_right(
            history,
            value,
        )

        denominator = len(history) + 1

        percentile[i] = 100.0 * position / denominator

        insort(
            history,
            value,
        )

        if (i + 1) % 100_000 == 0 or (i + 1) == total:
            print(f"  Percentile: {i + 1:,}/{total:,}")

    # -------------------------------------------------------------------------
    # Bucket.
    # -------------------------------------------------------------------------

    buckets = np.full(
        len(percentile),
        None,
        dtype=object,
    )

    finite = np.isfinite(percentile)

    buckets[finite & (percentile < 20)] = "0-20"

    buckets[finite & (percentile >= 20) & (percentile < 40)] = "20-40"

    buckets[finite & (percentile >= 40) & (percentile < 60)] = "40-60"

    buckets[finite & (percentile >= 60) & (percentile < 80)] = "60-80"

    buckets[finite & (percentile >= 80)] = "80-100"

    result = pd.Series(buckets)

    print()
    print("VOLATILITY BUCKET DISTRIBUTION")

    print(result.value_counts(dropna=False).sort_index())

    # -------------------------------------------------------------------------
    # Missing volatility should only occur at the initial undefined section.
    # -------------------------------------------------------------------------

    missing_mask = result.isna()

    missing_count = int(missing_mask.sum())

    if missing_count:
        print()
        print(f"Undefined initial volatility events: {missing_count:,}")

        invalid_indices = np.flatnonzero(missing_mask.to_numpy())

        expected = np.arange(missing_count)

        if not np.array_equal(
            invalid_indices,
            expected,
        ):
            raise RuntimeError(
                "Missing volatility observations occur "
                "outside the initial undefined region."
            )

    return result.reset_index(drop=True)


# =============================================================================
# Z-SCORE
# =============================================================================


def identify_zscore_column(metadata):

    candidates = [
        "zscore_30",
        "zscore",
        "z_score",
        "entry_zscore",
        "entry_z",
    ]

    for column in candidates:
        if column in metadata.columns:
            return column

    raise RuntimeError(
        f"Could not identify z-score column.\nColumns: {list(metadata.columns)}"
    )


def load_zscore(metadata):

    section("LOADING EVENT Z-SCORE")

    column = identify_zscore_column(metadata)

    zscore = pd.to_numeric(
        metadata[column],
        errors="coerce",
    )

    print(f"Using z-score column: {column}")

    print(f"Valid z-score rows: {zscore.notna().sum():,}")

    if zscore.isna().any():
        missing = int(zscore.isna().sum())

        print(f"Missing z-score rows: {missing:,}")

    return zscore.reset_index(drop=True)


# =============================================================================
# COMPLETE CONTEXT
# =============================================================================


def build_context(
    metadata,
    hmm,
    volatility,
    zscore,
):

    section("BUILDING COMPLETE EVENT CONTEXT")

    n = len(metadata)

    if len(hmm) != n:
        raise RuntimeError("HMM alignment failure.")

    if len(volatility) != n:
        raise RuntimeError("Volatility alignment failure.")

    if len(zscore) != n:
        raise RuntimeError("Z-score alignment failure.")

    context = pd.DataFrame(
        {
            "hmm_state": hmm["hmm_state"].to_numpy(),
            "vol_bucket": volatility.to_numpy(),
            "zscore": zscore.to_numpy(),
        }
    )

    print(f"Events: {len(context):,}")

    print(f"Missing HMM: {context['hmm_state'].isna().sum():,}")

    print(f"Missing volatility: {context['vol_bucket'].isna().sum():,}")

    print(f"Missing z-score: {context['zscore'].isna().sum():,}")

    # -------------------------------------------------------------------------
    # Remove only undefined initial volatility events.
    # -------------------------------------------------------------------------

    valid = (
        context["hmm_state"].notna()
        & context["vol_bucket"].notna()
        & context["zscore"].notna()
    )

    context = context[valid].reset_index(drop=True)

    print()
    print(f"Valid event contexts: {len(context):,}")

    return context


# =============================================================================
# CONTEXT INDICES
# =============================================================================


def get_context_indices(
    context,
    config,
):
    """
    Return event indices satisfying the frozen signal context.

    Mean-reversion z-score convention:
        LONG  -> zscore <= -threshold
        SHORT -> zscore >= +threshold

    The candidate zscore is therefore a magnitude/threshold,
    NOT an exact z-score value.
    """

    mask = context["hmm_state"].astype(int) == int(config["hmm_state"])

    mask &= context["vol_bucket"] == config["vol_bucket"]

    zscore = context["zscore"].astype(float)
    threshold = float(config["zscore"])

    if config["side"] == "LONG":
        mask &= zscore <= -threshold

    elif config["side"] == "SHORT":
        mask &= zscore >= threshold

    else:
        raise ValueError(f"Unknown side: {config['side']}")

    return np.flatnonzero(mask.to_numpy())


# =============================================================================
# PATH EVALUATION
# =============================================================================


def evaluate_paths(
    indices,
    arrays,
    side,
    tp,
    sl,
    horizon,
):

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if horizon < 1:
        raise ValueError("Horizon must be >= 1.")

    max_horizon = arrays["future_close"].shape[1]

    if horizon > max_horizon:
        raise ValueError(f"H={horizon} exceeds cache maximum H={max_horizon}.")

    if side == "LONG":
        favorable = arrays["long_favorable"][
            indices,
            :horizon,
        ]

        adverse = arrays["long_adverse"][
            indices,
            :horizon,
        ]

    elif side == "SHORT":
        favorable = arrays["short_favorable"][
            indices,
            :horizon,
        ]

        adverse = arrays["short_adverse"][
            indices,
            :horizon,
        ]

    else:
        raise ValueError(f"Unknown side: {side}")

    tp_hit = favorable >= (tp - EPS)

    sl_hit = adverse >= (sl - EPS)

    results = np.full(
        len(indices),
        "UNRESOLVED",
        dtype=object,
    )

    r = np.zeros(
        len(indices),
        dtype=float,
    )

    bars = np.full(
        len(indices),
        np.nan,
        dtype=float,
    )

    for i in range(len(indices)):
        tp_positions = np.flatnonzero(tp_hit[i])

        sl_positions = np.flatnonzero(sl_hit[i])

        first_tp = int(tp_positions[0]) if len(tp_positions) else None

        first_sl = int(sl_positions[0]) if len(sl_positions) else None

        # No threshold reached.
        if first_tp is None and first_sl is None:
            continue

        # Both reached on same bar.
        if first_tp is not None and first_sl is not None and first_tp == first_sl:
            results[i] = "AMBIGUOUS"

            bars[i] = first_tp + 1

            continue

        # TP first.
        if first_tp is not None and (first_sl is None or first_tp < first_sl):
            results[i] = "WIN"

            r[i] = tp / sl

            bars[i] = first_tp + 1

            continue

        # SL first.
        results[i] = "LOSS"

        r[i] = -1.0

        bars[i] = first_sl + 1

    return pd.DataFrame(
        {
            "event_index": indices,
            "result": results,
            "r": r,
            "bars_to_result": bars,
        }
    )


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(trade_results):

    n = len(trade_results)

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
            "net_r": np.nan,
            "expectancy_all": np.nan,
            "expectancy_resolved": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": np.nan,
        }

    win_mask = trade_results["result"] == "WIN"

    loss_mask = trade_results["result"] == "LOSS"

    ambiguous_mask = trade_results["result"] == "AMBIGUOUS"

    unresolved_mask = trade_results["result"] == "UNRESOLVED"

    resolved_mask = win_mask | loss_mask

    resolved = trade_results[resolved_mask]

    resolved_n = len(resolved)

    wins = int(win_mask.sum())

    losses = int(loss_mask.sum())

    ambiguous = int(ambiguous_mask.sum())

    unresolved = int(unresolved_mask.sum())

    if resolved_n:
        wr = wins / resolved_n

        resolution = resolved_n / n

        r_resolved = resolved["r"].astype(float)

        expectancy_resolved = float(r_resolved.mean())

        net_r = float(r_resolved.sum())

        gross_profit = float(r_resolved[r_resolved > 0].sum())

        gross_loss = float(abs(r_resolved[r_resolved < 0].sum()))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        equity = r_resolved.cumsum().to_numpy()

        running_max = np.maximum.accumulate(equity)

        drawdown = equity - running_max

        max_drawdown_r = float(drawdown.min())

    else:
        wr = np.nan
        resolution = 0.0
        expectancy_resolved = np.nan
        net_r = 0.0
        profit_factor = np.nan
        max_drawdown_r = np.nan

    expectancy_all = float(trade_results["r"].astype(float).mean())

    return {
        "observations": n,
        "wins": wins,
        "losses": losses,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
        "resolved": resolved_n,
        "wr": wr,
        "resolution": resolution,
        "net_r": net_r,
        "expectancy_all": expectancy_all,
        "expectancy_resolved": expectancy_resolved,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r,
    }


# =============================================================================
# PARAMETER VALUES
# =============================================================================


def get_parameter_values(config):

    tp_values = sorted(
        {
            max(
                0.1,
                config["tp"] - TP_STEP,
            ),
            config["tp"],
            config["tp"] + TP_STEP,
        }
    )

    sl_values = sorted(
        {
            max(
                0.1,
                config["sl"] - SL_STEP,
            ),
            config["sl"],
            config["sl"] + SL_STEP,
        }
    )

    h_values = sorted(
        {
            max(
                1,
                config["horizon"] - H_STEP,
            ),
            config["horizon"],
            config["horizon"] + H_STEP,
        }
    )

    return (
        tp_values,
        sl_values,
        h_values,
    )


# =============================================================================
# EVALUATE COMBINATION
# =============================================================================


def evaluate_combination(
    strategy,
    config,
    indices,
    arrays,
    tp,
    sl,
    horizon,
    surface,
):

    trade_results = evaluate_paths(
        indices=indices,
        arrays=arrays,
        side=config["side"],
        tp=tp,
        sl=sl,
        horizon=horizon,
    )

    m = calculate_metrics(trade_results)

    frozen = (
        np.isclose(
            tp,
            config["tp"],
            atol=1e-12,
        )
        and np.isclose(
            sl,
            config["sl"],
            atol=1e-12,
        )
        and int(horizon) == int(config["horizon"])
    )

    return {
        "strategy_name": strategy,
        "candidate_id": config["candidate_id"],
        "side": config["side"],
        "hmm_state": config["hmm_state"],
        "vol_bucket": config["vol_bucket"],
        "zscore": config["zscore"],
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(tp / sl),
        "horizon": int(horizon),
        "surface": surface,
        "is_frozen": frozen,
        **m,
    }


# =============================================================================
# NEIGHBORHOOD
# =============================================================================


def run_neighborhood(
    context,
    arrays,
):

    section("RUNNING TRUE PARAMETER NEIGHBORHOOD AUDIT")

    all_results = []

    for strategy, config in CANDIDATES.items():
        print()
        print("-" * 100)

        print(
            f"{strategy} | "
            f"{config['side']} | "
            f"HMM={config['hmm_state']} | "
            f"VOL={config['vol_bucket']} | "
            f"Z={config['zscore']} | "
            f"TP={config['tp']} | "
            f"SL={config['sl']} | "
            f"H={config['horizon']}"
        )

        indices = get_context_indices(
            context,
            config,
        )

        print(f"Context observations: {len(indices):,}")

        if len(indices) < MIN_OBSERVATIONS:
            print("SKIPPED — insufficient observations.")

            continue

        tp_values, sl_values, h_values = get_parameter_values(config)

        print(f"TP neighborhood: {tp_values}")

        print(f"SL neighborhood: {sl_values}")

        print(f"H neighborhood: {h_values}")

        # ---------------------------------------------------------------------
        # TP x SL @ frozen H
        # ---------------------------------------------------------------------

        for tp, sl in itertools.product(
            tp_values,
            sl_values,
        ):
            all_results.append(
                evaluate_combination(
                    strategy,
                    config,
                    indices,
                    arrays,
                    tp,
                    sl,
                    config["horizon"],
                    "TP_SL_at_frozen_H",
                )
            )

        # ---------------------------------------------------------------------
        # TP x H @ frozen SL
        # ---------------------------------------------------------------------

        for tp, horizon in itertools.product(
            tp_values,
            h_values,
        ):
            all_results.append(
                evaluate_combination(
                    strategy,
                    config,
                    indices,
                    arrays,
                    tp,
                    config["sl"],
                    horizon,
                    "TP_H_at_frozen_SL",
                )
            )

        # ---------------------------------------------------------------------
        # SL x H @ frozen TP
        # ---------------------------------------------------------------------

        for sl, horizon in itertools.product(
            sl_values,
            h_values,
        ):
            all_results.append(
                evaluate_combination(
                    strategy,
                    config,
                    indices,
                    arrays,
                    config["tp"],
                    sl,
                    horizon,
                    "SL_H_at_frozen_TP",
                )
            )

    if not all_results:
        raise RuntimeError("No neighborhood results were generated.")

    return pd.DataFrame(all_results)


# =============================================================================
# DISTANCE
# =============================================================================


def add_parameter_distance(results):

    output = results.copy()

    distance = []

    for _, row in output.iterrows():
        config = CANDIDATES[row["strategy_name"]]

        d_tp = abs(row["tp"] - config["tp"]) / TP_STEP

        d_sl = abs(row["sl"] - config["sl"]) / SL_STEP

        d_h = abs(row["horizon"] - config["horizon"]) / H_STEP

        distance.append(d_tp + d_sl + d_h)

    output["parameter_distance"] = distance

    return output


# =============================================================================
# STABILITY SUMMARY
# =============================================================================


def build_stability_summary(results):

    rows = []

    for strategy in CANDIDATES:
        group = results[results["strategy_name"] == strategy].copy()

        if group.empty:
            continue

        frozen = group[group["is_frozen"]]

        if frozen.empty:
            continue

        frozen_row = frozen.iloc[0]

        # All unique local combinations.
        local = group[group["observations"] >= MIN_OBSERVATIONS].drop_duplicates(
            subset=[
                "tp",
                "sl",
                "horizon",
            ]
        )

        positive_exp = local[local["expectancy_resolved"] > 0]

        pf_positive = local[local["profit_factor"] > 1]

        rows.append(
            {
                "strategy_name": strategy,
                "frozen_tp": frozen_row["tp"],
                "frozen_sl": frozen_row["sl"],
                "frozen_horizon": frozen_row["horizon"],
                "frozen_wr": frozen_row["wr"],
                "frozen_expectancy": frozen_row["expectancy_resolved"],
                "frozen_pf": frozen_row["profit_factor"],
                "local_points": len(local),
                "positive_expectancy_points": len(positive_exp),
                "positive_expectancy_ratio": (
                    len(positive_exp) / len(local) if len(local) else np.nan
                ),
                "pf_gt_1_points": len(pf_positive),
                "pf_gt_1_ratio": (
                    len(pf_positive) / len(local) if len(local) else np.nan
                ),
                "local_wr_min": local["wr"].min(),
                "local_wr_median": local["wr"].median(),
                "local_wr_max": local["wr"].max(),
                "local_expectancy_min": local["expectancy_resolved"].min(),
                "local_expectancy_median": local["expectancy_resolved"].median(),
                "local_expectancy_max": local["expectancy_resolved"].max(),
                "local_pf_min": local["profit_factor"]
                .replace(
                    [np.inf],
                    np.nan,
                )
                .min(),
                "local_pf_median": local["profit_factor"]
                .replace(
                    [np.inf],
                    np.nan,
                )
                .median(),
                "local_pf_max": local["profit_factor"]
                .replace(
                    [np.inf],
                    np.nan,
                )
                .max(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# FROZEN VS IMMEDIATE NEIGHBORS
# =============================================================================


def build_frozen_vs_neighbors(results):

    rows = []

    for strategy in CANDIDATES:
        group = results[results["strategy_name"] == strategy].copy()

        frozen = group[group["is_frozen"]]

        if frozen.empty:
            continue

        f = frozen.iloc[0]

        immediate = group[
            (group["parameter_distance"] > 0) & (group["parameter_distance"] <= 1)
        ].drop_duplicates(
            subset=[
                "tp",
                "sl",
                "horizon",
            ]
        )

        positive = immediate[immediate["expectancy_resolved"] > 0]

        rows.append(
            {
                "strategy_name": strategy,
                "frozen_wr": f["wr"],
                "frozen_expectancy": f["expectancy_resolved"],
                "frozen_pf": f["profit_factor"],
                "neighbor_count": len(immediate),
                "positive_neighbors": len(positive),
                "positive_neighbor_ratio": (
                    len(positive) / len(immediate) if len(immediate) else np.nan
                ),
                "neighbor_wr_mean": immediate["wr"].mean(),
                "neighbor_wr_min": immediate["wr"].min(),
                "neighbor_wr_max": immediate["wr"].max(),
                "neighbor_expectancy_mean": immediate["expectancy_resolved"].mean(),
                "neighbor_expectancy_min": immediate["expectancy_resolved"].min(),
                "neighbor_expectancy_max": immediate["expectancy_resolved"].max(),
                "neighbor_pf_mean": immediate["profit_factor"]
                .replace(
                    [np.inf],
                    np.nan,
                )
                .mean(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# HEATMAP GENERATOR
# =============================================================================


def save_heatmap(
    pivot,
    title,
    xlabel,
    ylabel,
    filename,
):

    if pivot.empty:
        return

    plt.figure(figsize=(10, 7))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    plt.colorbar(image, label="Expectancy (R)")

    plt.xticks(
        range(len(pivot.columns)),
        [f"{x:.2f}" for x in pivot.columns],
    )

    plt.yticks(
        range(len(pivot.index)),
        [f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}" for x in pivot.index],
    )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


def generate_heatmaps(results):

    section("GENERATING NEIGHBORHOOD HEATMAPS")

    for strategy, config in CANDIDATES.items():
        # ---------------------------------------------------------------------
        # TP x SL
        # ---------------------------------------------------------------------

        group = results[
            (results["strategy_name"] == strategy)
            & (results["surface"] == "TP_SL_at_frozen_H")
        ]

        if not group.empty:
            pivot = group.pivot(
                index="sl",
                columns="tp",
                values="expectancy_resolved",
            )

            save_heatmap(
                pivot,
                (f"{strategy} — TP / SL Neighborhood (H={config['horizon']})"),
                "TP",
                "SL",
                f"{strategy}_TP_SL_heatmap.png",
            )

        # ---------------------------------------------------------------------
        # TP x H
        # ---------------------------------------------------------------------

        group = results[
            (results["strategy_name"] == strategy)
            & (results["surface"] == "TP_H_at_frozen_SL")
        ]

        if not group.empty:
            pivot = group.pivot(
                index="horizon",
                columns="tp",
                values="expectancy_resolved",
            )

            save_heatmap(
                pivot,
                (f"{strategy} — TP / Horizon Neighborhood (SL={config['sl']})"),
                "TP",
                "Horizon",
                f"{strategy}_TP_H_heatmap.png",
            )

        # ---------------------------------------------------------------------
        # SL x H
        # ---------------------------------------------------------------------

        group = results[
            (results["strategy_name"] == strategy)
            & (results["surface"] == "SL_H_at_frozen_TP")
        ]

        if not group.empty:
            pivot = group.pivot(
                index="horizon",
                columns="sl",
                values="expectancy_resolved",
            )

            save_heatmap(
                pivot,
                (f"{strategy} — SL / Horizon Neighborhood (TP={config['tp']})"),
                "SL",
                "Horizon",
                f"{strategy}_SL_H_heatmap.png",
            )


# =============================================================================
# SENSITIVITY PLOTS
# =============================================================================


def generate_sensitivity_plots(results):

    section("GENERATING PARAMETER SENSITIVITY PLOTS")

    for strategy in CANDIDATES:
        group = results[results["strategy_name"] == strategy].copy()

        if group.empty:
            continue

        plt.figure(figsize=(12, 7))

        for surface in sorted(group["surface"].unique()):
            g = group[group["surface"] == surface].copy()

            # Aggregate combinations at identical parameter distance.
            g = (
                g.groupby(
                    "parameter_distance",
                    as_index=False,
                )["expectancy_resolved"]
                .mean()
                .sort_values("parameter_distance")
            )

            plt.plot(
                g["parameter_distance"],
                g["expectancy_resolved"],
                marker="o",
                label=surface,
            )

        plt.axhline(
            0,
            linewidth=1.0,
        )

        plt.title(f"{strategy} — Local Parameter Sensitivity")

        plt.xlabel("Distance from Frozen Point")

        plt.ylabel("Expectancy (R)")

        plt.grid(alpha=0.20)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_parameter_sensitivity.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.close()


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================


def print_summary(
    results,
    stability,
    comparison,
):

    section("08U SUMMARY")

    for strategy in CANDIDATES:
        print()
        print(strategy)

        group = results[results["strategy_name"] == strategy]

        frozen = group[group["is_frozen"]]

        if frozen.empty:
            print("No frozen result.")

            continue

        f = frozen.iloc[0]

        print(
            f"Frozen parameters: "
            f"TP={f['tp']:.2f} | "
            f"SL={f['sl']:.2f} | "
            f"H={int(f['horizon'])}"
        )

        print(
            f"N={int(f['observations']):,} | "
            f"WR={f['wr']:.2%} | "
            f"Resolution={f['resolution']:.2%}"
        )

        print(
            f"Expectancy={f['expectancy_resolved']:.4f}R | PF={f['profit_factor']:.4f}"
        )

        print(f"Net R={f['net_r']:.2f} | Max DD={f['max_drawdown_r']:.2f}R")

        if not stability.empty:
            s = stability[stability["strategy_name"] == strategy]

            if not s.empty:
                row = s.iloc[0]

                print()
                print("LOCAL ROBUSTNESS")

                print(
                    f"Positive expectancy: "
                    f"{row['positive_expectancy_points']}/"
                    f"{row['local_points']} "
                    f"({row['positive_expectancy_ratio']:.2%})"
                )

                print(
                    f"PF > 1: "
                    f"{row['pf_gt_1_points']}/"
                    f"{row['local_points']} "
                    f"({row['pf_gt_1_ratio']:.2%})"
                )

                print(
                    f"Local WR range: "
                    f"{row['local_wr_min']:.2%} "
                    f"→ "
                    f"{row['local_wr_max']:.2%}"
                )

                print(
                    f"Local expectancy range: "
                    f"{row['local_expectancy_min']:.4f}R "
                    f"→ "
                    f"{row['local_expectancy_max']:.4f}R"
                )

                print(
                    f"Local PF range: "
                    f"{row['local_pf_min']:.4f} "
                    f"→ "
                    f"{row['local_pf_max']:.4f}"
                )

        if not comparison.empty:
            c = comparison[comparison["strategy_name"] == strategy]

            if not c.empty:
                row = c.iloc[0]

                print()
                print("IMMEDIATE NEIGHBORS")

                print(f"Neighbors: {int(row['neighbor_count'])}")

                print(
                    f"Positive neighbors: "
                    f"{int(row['positive_neighbors'])} "
                    f"({row['positive_neighbor_ratio']:.2%})"
                )

                print(
                    f"Neighbor expectancy: "
                    f"{row['neighbor_expectancy_min']:.4f}R "
                    f"→ "
                    f"{row['neighbor_expectancy_max']:.4f}R"
                )


# =============================================================================
# REPORT
# =============================================================================


def write_report(
    stability,
    comparison,
):

    report_path = OUT_DIR / "research_08u_parameter_neighborhood_report.txt"

    lines = []

    lines.append("MEAN REVERSION — RESEARCH 08U")

    lines.append("TRUE PARAMETER NEIGHBORHOOD ROBUSTNESS")

    lines.append("")

    lines.append("Frozen candidates were not changed.")

    lines.append("No optimization was performed.")

    lines.append("Research 07 path cache was evaluated directly.")

    lines.append("")

    lines.append("FROZEN CANDIDATES")

    for strategy, config in CANDIDATES.items():
        lines.append(
            f"{strategy}: "
            f"{config['side']} | "
            f"HMM={config['hmm_state']} | "
            f"VOL={config['vol_bucket']} | "
            f"Z={config['zscore']} | "
            f"TP={config['tp']} | "
            f"SL={config['sl']} | "
            f"H={config['horizon']}"
        )

    lines.append("")

    lines.append("STABILITY SUMMARY")

    if stability.empty:
        lines.append("No stability results.")

    else:
        lines.append(stability.to_string(index=False))

    lines.append("")

    lines.append("FROZEN VS IMMEDIATE NEIGHBORS")

    if comparison.empty:
        lines.append("No comparison results.")

    else:
        lines.append(comparison.to_string(index=False))

    lines.append("")

    lines.append("INTERPRETATION")

    lines.append("The best neighbor is NOT selected.")

    lines.append(
        "A neighbor with higher expectancy does NOT replace the frozen candidate."
    )

    lines.append(
        "The purpose is to determine whether the frozen "
        "point is an isolated peak or part of a robust region."
    )

    lines.append("")

    lines.append("NEXT RESEARCH STAGE")

    lines.append(
        "Combine neighborhood robustness with temporal OOS, "
        "failure analysis, Monte Carlo and final validation."
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08U")

    print("TRUE PARAMETER NEIGHBORHOOD ROBUSTNESS")

    print("-" * 100)

    print("Frozen candidates:")

    for strategy, config in CANDIDATES.items():
        print(
            f"{strategy}: "
            f"{config['side']} | "
            f"HMM={config['hmm_state']} | "
            f"VOL={config['vol_bucket']} | "
            f"Z={config['zscore']} | "
            f"TP={config['tp']} | "
            f"SL={config['sl']} | "
            f"H={config['horizon']}"
        )

    print()
    print("NO optimization.")
    print("NO candidate replacement.")
    print("NO HMM retraining.")
    print("NO volatility optimization.")

    create_directories()

    # =========================================================================
    # LOAD BASE DATA
    # =========================================================================

    metadata, arrays = load_research_07()

    hmm = load_hmm()

    # =========================================================================
    # STRICT ROW ALIGNMENT
    # =========================================================================

    if len(hmm) != len(metadata):
        raise RuntimeError("HMM and Research 07 metadata are not aligned.")

    # =========================================================================
    # BUILD CAUSAL VOLATILITY
    # =========================================================================

    volatility = build_volatility_context(metadata)

    zscore = load_zscore(metadata)

    # =========================================================================
    # COMPLETE CONTEXT
    # =========================================================================

    context = build_context(
        metadata,
        hmm,
        volatility,
        zscore,
    )

    context_path = TABLES_DIR / "research_08u_event_context.csv"

    context.to_csv(
        context_path,
        index=False,
    )

    # =========================================================================
    # RUN AUDIT
    # =========================================================================

    results = run_neighborhood(
        context,
        arrays,
    )

    results = add_parameter_distance(results)

    # =========================================================================
    # SAVE RAW RESULTS
    # =========================================================================

    raw_path = TABLES_DIR / "research_08u_all_parameter_results.csv"

    results.to_csv(
        raw_path,
        index=False,
    )

    # =========================================================================
    # STABILITY
    # =========================================================================

    stability = build_stability_summary(results)

    stability_path = TABLES_DIR / "research_08u_stability_summary.csv"

    stability.to_csv(
        stability_path,
        index=False,
    )

    # =========================================================================
    # FROZEN VS NEIGHBORS
    # =========================================================================

    comparison = build_frozen_vs_neighbors(results)

    comparison_path = TABLES_DIR / "research_08u_frozen_vs_neighbors.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # =========================================================================
    # PLOTS
    # =========================================================================

    generate_heatmaps(results)

    generate_sensitivity_plots(results)

    # =========================================================================
    # REPORT
    # =========================================================================

    report_path = write_report(
        stability,
        comparison,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_summary(
        results,
        stability,
        comparison,
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    section("RESEARCH 08U COMPLETE")

    print("TRUE parameter neighborhood audit completed.")

    print()
    print("Tables:")

    print(f"  {context_path}")

    print(f"  {raw_path}")

    print(f"  {stability_path}")

    print(f"  {comparison_path}")

    print()
    print("Plots:")

    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {path}")

    print()
    print("Report:")

    print(f"  {report_path}")

    print()
    print("IMPORTANT:")

    print("This research does NOT select a new parameter set.")

    print("It only tests whether the frozen candidates are locally robust.")


if __name__ == "__main__":
    main()
