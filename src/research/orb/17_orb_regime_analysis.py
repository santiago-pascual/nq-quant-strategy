from __future__ import annotations

import sys
from pathlib import Path

# =============================================================================
# PROJECT ROOT / PYTHON PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import pandas as pd

from src.databento_loader import load_databento_mnq
from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)


# =============================================================================
# ORB REGIME ANALYSIS
# =============================================================================
#
# Purpose:
#
#   Determine where the frozen ORB strategy operates in the same
#   causal HMM + volatility regime framework used by the MR system.
#
# IMPORTANT:
#
#   - ORB parameters are frozen.
#   - External OOS window is frozen.
#   - No optimization is performed.
#   - No trades are removed based on performance.
#   - HMM states come from the frozen causal HMM cache.
#   - realized_vol_30 comes from the canonical market pipeline.
#   - VOL percentile/bucket uses the same causal methodology as MR.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results"

ORB_RESULTS_DIR = RESULTS_DIR / "orb"

CONTEXT_DIR = ORB_RESULTS_DIR / "context_analysis"

MEAN_REVERSION_RESULTS = (
    PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"
)

CACHE_DIR = MEAN_REVERSION_RESULTS / "cache"

ORB_TRADES = ORB_RESULTS_DIR / "orb_reconciliation_trades.csv"

HMM_CACHE = CACHE_DIR / "research_08b_causal_hmm_states.csv"

RESEARCH_07_METADATA = CACHE_DIR / "research_07_event_metadata.csv"


# =============================================================================
# EXTERNAL OOS WINDOW
# =============================================================================

OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="America/New_York",
)

OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59",
    tz="America/New_York",
)


# =============================================================================
# VOLATILITY BUCKETS
# =============================================================================

VOL_BUCKETS = [
    (0.0, 20.0, "VOL0-20"),
    (20.0, 40.0, "VOL20-40"),
    (40.0, 60.0, "VOL40-60"),
    (60.0, 80.0, "VOL60-80"),
    (80.0, 100.0, "VOL80-100"),
]


# =============================================================================
# HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to America/New_York and force nanosecond
    precision so pandas merge_asof receives identical datetime dtypes.
    """
    ts = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    ts = ts.dt.tz_convert("America/New_York")

    # Force identical datetime64[ns, America/New_York] dtype
    return ts.astype("datetime64[ns, America/New_York]")


def classify_vol_bucket(percentile: float) -> str:
    if pd.isna(percentile):
        return "UNKNOWN"

    if percentile < 20:
        return "VOL0-20"

    if percentile < 40:
        return "VOL20-40"

    if percentile < 60:
        return "VOL40-60"

    if percentile < 80:
        return "VOL60-80"

    return "VOL80-100"


def causal_percentile(series: pd.Series) -> pd.Series:
    """
    Causal expanding percentile.

    For observation i, percentile is computed only using observations
    available before i.

    This matches the causal philosophy used by the MR pipeline.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    history: list[float] = []

    for i, value in enumerate(values.to_numpy(dtype=float)):
        if np.isfinite(value):
            if history:
                arr = np.asarray(
                    history,
                    dtype=float,
                )

                result[i] = np.sum(arr <= value) / len(arr) * 100.0

            history.append(float(value))

    return pd.Series(
        result,
        index=series.index,
        name="vol_percentile",
    )


def calculate_stats(df: pd.DataFrame) -> dict:

    if len(df) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "total_r": 0.0,
            "expectancy_r": np.nan,
            "pf": np.nan,
            "avg_win_r": np.nan,
            "avg_loss_r": np.nan,
        }

    # ORB frozen trade file stores R-multiple as `net_R`.
    r = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    ).dropna()

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "trades": int(len(r)),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate": float((r > 0).mean()),
        "total_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "pf": float(pf),
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses.mean()) if len(losses) else np.nan,
    }

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "trades": int(len(r)),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate": float((r > 0).mean()),
        "total_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "pf": float(pf),
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses.mean()) if len(losses) else np.nan,
    }


# =============================================================================
# LOAD ORB
# =============================================================================


banner("LOAD FROZEN ORB TRADES")

if not ORB_TRADES.exists():
    raise FileNotFoundError(f"ORB trade file not found:\n{ORB_TRADES}")

trades = pd.read_csv(ORB_TRADES)

print(f"ORB rows: {len(trades):,}")
print(f"ORB columns: {list(trades.columns)}")


# =============================================================================
# NORMALIZE ORB TIMESTAMP
# =============================================================================


timestamp_candidates = [
    "entry_timestamp",
    "timestamp",
    "entry_time",
]

orb_timestamp_column = None

for column in timestamp_candidates:
    if column in trades.columns:
        orb_timestamp_column = column
        break

if orb_timestamp_column is None:
    raise ValueError("Could not identify ORB entry timestamp.")

trades["entry_timestamp_et"] = normalize_timestamp(trades[orb_timestamp_column])

trades = trades.dropna(subset=["entry_timestamp_et"]).copy()


# =============================================================================
# EXACT EXTERNAL OOS
# =============================================================================


banner("FILTER EXACT EXTERNAL OOS")

orb_oos = trades[
    (trades["entry_timestamp_et"] >= OOS_START)
    & (trades["entry_timestamp_et"] <= OOS_END)
].copy()

orb_oos = orb_oos.sort_values("entry_timestamp_et").reset_index(drop=True)

print(f"OOS window: {OOS_START} -> {OOS_END}")

print(f"OOS trades: {len(orb_oos):,}")


if len(orb_oos) == 0:
    raise RuntimeError("No ORB trades inside external OOS window.")


# =============================================================================
# LOAD FROZEN HMM
# =============================================================================


banner("LOAD FROZEN CAUSAL HMM")

if not HMM_CACHE.exists():
    raise FileNotFoundError(f"HMM cache not found:\n{HMM_CACHE}")

hmm = pd.read_csv(HMM_CACHE)

print(f"HMM rows: {len(hmm):,}")
print(f"HMM columns: {list(hmm.columns)}")


if "hmm_state" not in hmm.columns:
    raise ValueError("Frozen HMM cache does not contain 'hmm_state'.")


hmm_timestamp_candidates = [
    "timestamp",
    "timestamp ET",
    "ts_event",
]

hmm_timestamp_column = None

for column in hmm_timestamp_candidates:
    if column in hmm.columns:
        hmm_timestamp_column = column
        break

if hmm_timestamp_column is None:
    raise ValueError("Could not identify HMM timestamp.")

hmm["timestamp_et"] = normalize_timestamp(hmm[hmm_timestamp_column])

hmm["hmm_state"] = pd.to_numeric(
    hmm["hmm_state"],
    errors="coerce",
)

hmm = hmm[
    [
        "timestamp_et",
        "hmm_state",
    ]
].dropna(
    subset=[
        "timestamp_et",
        "hmm_state",
    ]
)

hmm = (
    hmm.sort_values("timestamp_et")
    .drop_duplicates(
        subset="timestamp_et",
        keep="last",
    )
    .reset_index(drop=True)
)

print(f"Valid HMM rows: {len(hmm):,}")


# =============================================================================
# LOAD CANONICAL MARKET DATA
# =============================================================================


banner("LOAD CANONICAL MARKET DATA")

market = load_databento_mnq()

print(f"Market rows: {len(market):,}")
print(f"Market columns: {list(market.columns)}")


# =============================================================================
# NORMALIZE MARKET TIMESTAMP
# =============================================================================


market_timestamp_candidates = [
    "timestamp",
    "timestamp ET",
    "ts_event",
]

market_timestamp_column = None

for column in market_timestamp_candidates:
    if column in market.columns:
        market_timestamp_column = column
        break

if market_timestamp_column is None:
    raise RuntimeError("Could not identify canonical market timestamp.")

print(f"Using market timestamp: {market_timestamp_column}")

market["_timestamp_17"] = normalize_timestamp(market[market_timestamp_column])


# =============================================================================
# BUILD EXACT CANONICAL VOLATILITY PIPELINE
# =============================================================================


banner("BUILD CANONICAL REALIZED VOLATILITY")

print("Building return features...")

market = add_return_features(market)

print("Building volatility features...")

market = add_volatility_features(market)

if "realized_vol_30" not in market.columns:
    raise RuntimeError("Canonical volatility pipeline did not produce realized_vol_30.")

market["realized_vol_30"] = pd.to_numeric(
    market["realized_vol_30"],
    errors="coerce",
)

print(f"Valid realized_vol_30: {market['realized_vol_30'].notna().sum():,}")

print(f"Missing realized_vol_30: {market['realized_vol_30'].isna().sum():,}")


# =============================================================================
# KEEP ONLY NECESSARY MARKET CONTEXT
# =============================================================================


market = market[
    [
        "_timestamp_17",
        "realized_vol_30",
    ]
].dropna(
    subset=[
        "_timestamp_17",
        "realized_vol_30",
    ]
)

market = (
    market.sort_values("_timestamp_17")
    .drop_duplicates(
        subset="_timestamp_17",
        keep="last",
    )
    .reset_index(drop=True)
)

print(f"Market volatility observations: {len(market):,}")


# =============================================================================
# MAP HMM STATE TO ORB ENTRIES
# =============================================================================


banner("MAP FROZEN HMM STATE")

print(
    "ORB timestamp dtype:",
    orb_oos["entry_timestamp_et"].dtype,
)

print(
    "Market timestamp dtype:",
    market["_timestamp_17"].dtype,
)

if orb_oos["entry_timestamp_et"].dtype != market["_timestamp_17"].dtype:
    raise RuntimeError("Timestamp dtype mismatch before realized-vol merge.")

orb_oos = pd.merge_asof(
    orb_oos.sort_values("entry_timestamp_et"),
    hmm.sort_values("timestamp_et"),
    left_on="entry_timestamp_et",
    right_on="timestamp_et",
    direction="backward",
    allow_exact_matches=True,
)

hmm_missing = int(orb_oos["hmm_state"].isna().sum())

print(f"HMM missing: {hmm_missing:,}/{len(orb_oos):,}")

if hmm_missing > 0:
    raise RuntimeError("Some ORB trades could not be mapped to a frozen HMM state.")


# =============================================================================
# MAP CAUSAL REALIZED VOLATILITY
# =============================================================================


banner("MAP CAUSAL REALIZED VOLATILITY")

#
# IMPORTANT:
#
# This is deliberately the same causal mechanism as Research 08H:
#
#     direction="backward"
#     allow_exact_matches=True
#
# Therefore each ORB entry receives the most recent canonical
# realized_vol_30 observation at or before entry.
#

orb_oos = pd.merge_asof(
    orb_oos.sort_values("entry_timestamp_et"),
    market.sort_values("_timestamp_17"),
    left_on="entry_timestamp_et",
    right_on="_timestamp_17",
    direction="backward",
    allow_exact_matches=True,
)

vol_missing = int(orb_oos["realized_vol_30"].isna().sum())

print(f"VOL missing: {vol_missing:,}/{len(orb_oos):,}")

if vol_missing > 0:
    raise RuntimeError("Some ORB trades could not be mapped to realized_vol_30.")


# =============================================================================
# CAUSAL VOLATILITY PERCENTILE
# =============================================================================


banner("BUILD CAUSAL VOLATILITY PERCENTILE")

#
# The percentile is calculated chronologically over the mapped
# ORB observations only.
#
# No future ORB trade information is used for a given observation.
#

orb_oos = orb_oos.sort_values("entry_timestamp_et").reset_index(drop=True)

orb_oos["vol_percentile"] = causal_percentile(orb_oos["realized_vol_30"])

orb_oos["vol_bucket"] = orb_oos["vol_percentile"].apply(classify_vol_bucket)


# =============================================================================
# BASIC VALIDATION
# =============================================================================


banner("REGIME MAPPING VALIDATION")

print(f"ORB OOS trades: {len(orb_oos):,}")

print("HMM states:")

print(orb_oos["hmm_state"].value_counts().sort_index().to_string())

print()

print("VOL buckets:")

print(
    orb_oos["vol_bucket"]
    .value_counts()
    .reindex(
        [
            "VOL0-20",
            "VOL20-40",
            "VOL40-60",
            "VOL60-80",
            "VOL80-100",
        ]
    )
    .fillna(0)
    .astype(int)
    .to_string()
)

print()

print(
    "realized_vol_30 range: "
    f"{orb_oos['realized_vol_30'].min():.8f}"
    " -> "
    f"{orb_oos['realized_vol_30'].max():.8f}"
)


# =============================================================================
# ORB BY HMM STATE
# =============================================================================


banner("ORB PERFORMANCE BY HMM STATE")

hmm_rows = []

for state, group in orb_oos.groupby("hmm_state", sort=True):
    stats = calculate_stats(group)

    stats["hmm_state"] = int(state)

    hmm_rows.append(stats)

hmm_summary = pd.DataFrame(hmm_rows)

print(hmm_summary.to_string(index=False))


# =============================================================================
# ORB BY VOL BUCKET
# =============================================================================


banner("ORB PERFORMANCE BY VOL BUCKET")

vol_rows = []

ordered_buckets = [
    "VOL0-20",
    "VOL20-40",
    "VOL40-60",
    "VOL60-80",
    "VOL80-100",
]

for bucket in ordered_buckets:
    group = orb_oos[orb_oos["vol_bucket"] == bucket]

    stats = calculate_stats(group)

    stats["vol_bucket"] = bucket

    vol_rows.append(stats)

vol_summary = pd.DataFrame(vol_rows)

print(vol_summary.to_string(index=False))


# =============================================================================
# ORB BY HMM × VOL
# =============================================================================


banner("ORB PERFORMANCE BY HMM × VOL")

regime_rows = []

for (state, bucket), group in orb_oos.groupby(
    ["hmm_state", "vol_bucket"],
    sort=True,
):
    stats = calculate_stats(group)

    stats["hmm_state"] = int(state)
    stats["vol_bucket"] = bucket

    regime_rows.append(stats)

regime_summary = pd.DataFrame(regime_rows)[
    [
        "hmm_state",
        "vol_bucket",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "total_r",
        "expectancy_r",
        "pf",
        "avg_win_r",
        "avg_loss_r",
    ]
]

print(regime_summary.to_string(index=False))


# =============================================================================
# COMPARE ORB REGIMES TO EXISTING MR MAP
# =============================================================================


banner("COMPARE WITH EXISTING MR REGIME MAP")

print(
    """
Existing frozen MR map:

    VOL0-20   -> uncovered
    VOL20-40  -> MRL1
    VOL40-60  -> S2R
    VOL60-80  -> uncovered
    VOL80-100 -> MRS2

This analysis does NOT decide whether ORB should be added.
It only measures where the frozen ORB historically traded
and how it performed in the same regime coordinates.
"""
)

comparison_rows = []

for _, row in regime_summary.iterrows():
    state = int(row["hmm_state"])
    bucket = row["vol_bucket"]

    if bucket == "VOL20-40":
        mr_strategy = "MRL1"

    elif bucket == "VOL40-60":
        mr_strategy = "S2R"

    elif bucket == "VOL80-100":
        mr_strategy = "MRS2"

    else:
        mr_strategy = "UNCOVERED"

    comparison_rows.append(
        {
            "hmm_state": state,
            "vol_bucket": bucket,
            "mr_strategy": mr_strategy,
            "orb_trades": int(row["trades"]),
            "orb_total_r": row["total_r"],
            "orb_expectancy_r": row["expectancy_r"],
            "orb_pf": row["pf"],
            "orb_win_rate": row["win_rate"],
        }
    )

comparison = pd.DataFrame(comparison_rows)

print(comparison.to_string(index=False))


# =============================================================================
# YEARLY REGIME STABILITY
# =============================================================================


banner("ORB HMM × VOL YEARLY STABILITY")

orb_oos["year"] = orb_oos["entry_timestamp_et"].dt.year

year_regime_rows = []

for (
    year,
    state,
    bucket,
), group in orb_oos.groupby(
    [
        "year",
        "hmm_state",
        "vol_bucket",
    ],
    sort=True,
):
    stats = calculate_stats(group)

    year_regime_rows.append(
        {
            "year": int(year),
            "hmm_state": int(state),
            "vol_bucket": bucket,
            **stats,
        }
    )

year_regime_summary = pd.DataFrame(year_regime_rows)


# =============================================================================
# SAVE RESULTS
# =============================================================================


banner("SAVE RESULTS")

CONTEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ORB_ENRICHED = CONTEXT_DIR / "orb_oos_regime_enriched.csv"

HMM_OUTPUT = CONTEXT_DIR / "orb_oos_hmm_regime.csv"

VOL_OUTPUT = CONTEXT_DIR / "orb_oos_vol_regime.csv"

REGIME_OUTPUT = CONTEXT_DIR / "orb_oos_hmm_x_vol_regime.csv"

COMPARISON_OUTPUT = CONTEXT_DIR / "orb_vs_mr_regime_map.csv"

YEAR_OUTPUT = CONTEXT_DIR / "orb_oos_hmm_x_vol_yearly.csv"

orb_oos.to_csv(
    ORB_ENRICHED,
    index=False,
)

hmm_summary.to_csv(
    HMM_OUTPUT,
    index=False,
)

vol_summary.to_csv(
    VOL_OUTPUT,
    index=False,
)

regime_summary.to_csv(
    REGIME_OUTPUT,
    index=False,
)

comparison.to_csv(
    COMPARISON_OUTPUT,
    index=False,
)

year_regime_summary.to_csv(
    YEAR_OUTPUT,
    index=False,
)


# =============================================================================
# FINAL AUDIT
# =============================================================================


banner("FINAL AUDIT")

print(f"ORB OOS trades:              {len(orb_oos):,}")

print(f"Unique ORB entry timestamps: {orb_oos['entry_timestamp_et'].nunique():,}")

print(f"HMM missing:                 {orb_oos['hmm_state'].isna().sum():,}")

print(f"VOL missing:                 {orb_oos['realized_vol_30'].isna().sum():,}")

print(f"VOL percentile missing:      {orb_oos['vol_percentile'].isna().sum():,}")

print("Parameters optimized:        NO")

print("Trades removed by performance: NO")

print()
print("OUTPUTS:")
print(ORB_ENRICHED)
print(HMM_OUTPUT)
print(VOL_OUTPUT)
print(REGIME_OUTPUT)
print(COMPARISON_OUTPUT)
print(YEAR_OUTPUT)

print()
print("=" * 80)
print("ORB REGIME ANALYSIS COMPLETE")
print("=" * 80)
