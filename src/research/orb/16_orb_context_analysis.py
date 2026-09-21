"""
16_orb_context_analysis.py

ORB Context Analysis
====================

Purpose
-------
Analyze the exact external OOS ORB sample without optimizing the strategy.

Frozen ORB specification:
    Opening Range: 30 minutes
    OR: 09:30 -> 10:00 ET
    Breakout: touch of OR high / OR low
    Entry: breakout level
    Stop: opposite side of OR
    TP: 2R
    One trade per RTH session
    No new entries after 11:00 ET
    Position may remain open until SL / TP / RTH close

External OOS:
    2020-06-23 -> 2026-06-19

This script DOES NOT:
    - optimize OR length
    - optimize RR
    - optimize cutoff
    - optimize filters
    - select a "best" regime
    - modify the frozen ORB trades

It only describes the already-frozen OOS trades in different market contexts.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb"

ORB_TRADES = RESULTS_DIR / "orb_reconciliation_trades.csv"

OUTPUT_DIR = RESULTS_DIR / "context_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="America/New_York",
)

OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59",
    tz="America/New_York",
)


# ============================================================
# HELPERS
# ============================================================


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Normalize mixed timestamp representations to America/New_York.

    The ORB reconciliation CSV may contain timestamps with
    different explicit timezone offsets, so parsing through UTC
    first is required.
    """
    ts = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return ts.dt.tz_convert("America/New_York")


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calc_pf(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss <= 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def summarize(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:

    if group_cols is None:
        group_cols = []

    work = df.copy()

    if "R" not in work.columns:
        raise ValueError("Missing R column.")

    work["R"] = pd.to_numeric(work["R"], errors="coerce")

    def agg_func(g: pd.DataFrame) -> pd.Series:
        r = g["R"].dropna()

        wins = (r > 0).sum()
        losses = (r < 0).sum()

        return pd.Series(
            {
                "trades": len(r),
                "wins": wins,
                "losses": losses,
                "win_rate": (wins / len(r) if len(r) else np.nan),
                "total_R": r.sum(),
                "expectancy_R": r.mean(),
                "median_R": r.median(),
                "PF": calc_pf(r),
                "avg_win_R": (r[r > 0].mean() if wins else np.nan),
                "avg_loss_R": (r[r < 0].mean() if losses else np.nan),
            }
        )

    if group_cols:
        out = (
            work.groupby(group_cols, dropna=False)
            .apply(agg_func, include_groups=False)
            .reset_index()
        )
    else:
        out = agg_func(work).to_frame().T

    return out


# ============================================================
# LOAD FROZEN ORB TRADES
# ============================================================

section("LOAD FROZEN ORB OOS TRADES")

if not ORB_TRADES.exists():
    raise FileNotFoundError(f"ORB reconciliation file not found:\n{ORB_TRADES}")

trades = pd.read_csv(ORB_TRADES)

# The frozen ORB reconciliation file calls the realized
# trade return "net_R".
if "R" not in trades.columns and "net_R" in trades.columns:
    trades["R"] = pd.to_numeric(
        trades["net_R"],
        errors="coerce",
    )

print(f"Loaded rows: {len(trades):,}")
print(f"Columns: {list(trades.columns)}")


# ============================================================
# NORMALIZE CORE COLUMNS
# ============================================================

timestamp_candidates = [
    "entry_timestamp",
    "entry_time",
    "timestamp",
    "entry_datetime",
]

entry_col = next(
    (c for c in timestamp_candidates if c in trades.columns),
    None,
)

if entry_col is None:
    raise ValueError(
        "Could not find entry timestamp column. "
        f"Available columns: {list(trades.columns)}"
    )

trades["entry_timestamp"] = normalize_timestamp(trades[entry_col])


# Exit timestamp

exit_candidates = [
    "exit_timestamp",
    "exit_time",
    "exit_datetime",
]

exit_col = next(
    (c for c in exit_candidates if c in trades.columns),
    None,
)

if exit_col is not None:
    trades["exit_timestamp"] = normalize_timestamp(trades[exit_col])
else:
    trades["exit_timestamp"] = pd.NaT


# Numeric columns

trades = safe_numeric(
    trades,
    [
        "R",
        "entry_price",
        "exit_price",
        "or_high",
        "or_low",
        "or_width",
        "holding_minutes",
        "holding_bars",
        "hmm_state",
        "realized_vol_30",
        "vol_percentile",
        "vol_pct",
        "quality",
    ],
)


# ============================================================
# EXACT OOS FILTER
# ============================================================

trades = trades[
    (trades["entry_timestamp"] >= OOS_START) & (trades["entry_timestamp"] <= OOS_END)
].copy()

trades = trades.sort_values("entry_timestamp").reset_index(drop=True)


section("OOS SAMPLE CHECK")

print(f"OOS start: {trades['entry_timestamp'].min()}")
print(f"OOS end:   {trades['entry_timestamp'].max()}")
print(f"Trades:    {len(trades):,}")


# ============================================================
# DERIVE BASIC CONTEXT
# ============================================================

section("DERIVE BASIC CONTEXT")


# Direction

if "side" in trades.columns:
    trades["direction"] = trades["side"].astype(str).str.upper()
elif "direction" in trades.columns:
    trades["direction"] = trades["direction"].astype(str).str.upper()
else:
    trades["direction"] = "UNKNOWN"


# Weekday

trades["weekday"] = trades["entry_timestamp"].dt.day_name()


trades["weekday_num"] = trades["entry_timestamp"].dt.weekday


# Entry hour/minute

trades["entry_hour"] = trades["entry_timestamp"].dt.hour

trades["entry_minute"] = trades["entry_timestamp"].dt.minute


trades["entry_time"] = trades["entry_timestamp"].dt.strftime("%H:%M")


# ============================================================
# BREAKOUT TIME
# ============================================================

if "entry_timestamp" in trades.columns:
    trades["breakout_minutes_after_10"] = (
        trades["entry_timestamp"].dt.hour * 60 + trades["entry_timestamp"].dt.minute
    ) - (10 * 60)


# ============================================================
# HOLDING TIME
# ============================================================

# Prefer the frozen holding time already stored in the
# reconciliation file. Only derive it when necessary.
if "holding_minutes" in trades.columns:
    trades["holding_minutes"] = pd.to_numeric(
        trades["holding_minutes"],
        errors="coerce",
    )

elif trades["exit_timestamp"].notna().any():
    holding = trades["exit_timestamp"] - trades["entry_timestamp"]

    trades["holding_minutes"] = holding.dt.total_seconds() / 60.0

# Holding buckets

if "holding_minutes" in trades.columns:
    trades["holding_bucket"] = pd.cut(
        trades["holding_minutes"],
        bins=[
            -np.inf,
            30,
            60,
            120,
            240,
            360,
            480,
            np.inf,
        ],
        labels=[
            "<=30m",
            "31-60m",
            "61-120m",
            "121-240m",
            "241-360m",
            "361-480m",
            ">480m",
        ],
    )


# ============================================================
# EXIT REASON
# ============================================================

if "exit_reason" in trades.columns:
    trades["exit_reason_clean"] = (
        trades["exit_reason"].astype(str).str.upper().str.strip()
    )

elif "exit_type" in trades.columns:
    trades["exit_reason_clean"] = (
        trades["exit_type"].astype(str).str.upper().str.strip()
    )

else:
    trades["exit_reason_clean"] = "UNKNOWN"


# ============================================================
# OR WIDTH
# ============================================================

# The frozen ORB file stores the opening-range width
# directly as "or_range".
if "or_range" in trades.columns:
    trades["or_width_points"] = pd.to_numeric(
        trades["or_range"],
        errors="coerce",
    )

elif "or_width" in trades.columns:
    trades["or_width_points"] = pd.to_numeric(
        trades["or_width"],
        errors="coerce",
    )

elif "or_high" in trades.columns and "or_low" in trades.columns:
    trades["or_width_points"] = pd.to_numeric(
        trades["or_high"], errors="coerce"
    ) - pd.to_numeric(trades["or_low"], errors="coerce")

else:
    trades["or_width_points"] = np.nan


if trades["or_width_points"].notna().any():
    trades["or_width_bucket"] = pd.qcut(
        trades["or_width_points"],
        q=5,
        duplicates="drop",
    )


# ============================================================
# VOL REGIME
# ============================================================

if "realized_vol_30" in trades.columns:
    # Match the existing research convention:
    # percentile-style 0-100 buckets.

    vol = pd.to_numeric(
        trades["realized_vol_30"],
        errors="coerce",
    )

    if vol.notna().sum() > 0:
        trades["vol_percentile_derived"] = (
            vol.rank(
                pct=True,
                method="average",
            )
            * 100.0
        )

        trades["vol_bucket"] = pd.cut(
            trades["vol_percentile_derived"],
            bins=[
                -np.inf,
                20,
                40,
                60,
                80,
                np.inf,
            ],
            labels=[
                "VOL0-20",
                "VOL20-40",
                "VOL40-60",
                "VOL60-80",
                "VOL80-100",
            ],
            include_lowest=True,
        )


# ============================================================
# HMM CONTEXT
# ============================================================

if "hmm_state" not in trades.columns:
    print()
    print("WARNING: frozen ORB trade file does not contain 'hmm_state'.")

    print(
        "HMM analysis will be skipped rather than "
        "inventing or reconstructing state labels."
    )


# ============================================================
# CORE SUMMARY
# ============================================================

section("CORE OOS SUMMARY")

core = summarize(trades)

print(core.to_string(index=False))

core.to_csv(
    OUTPUT_DIR / "orb_oos_core_summary.csv",
    index=False,
)


# ============================================================
# DIRECTION
# ============================================================

section("DIRECTION")

if "direction" in trades.columns:
    direction_summary = summarize(
        trades,
        ["direction"],
    )

    print(direction_summary.to_string(index=False))

    direction_summary.to_csv(
        OUTPUT_DIR / "orb_oos_direction.csv",
        index=False,
    )


# ============================================================
# WEEKDAY
# ============================================================

section("WEEKDAY")

weekday_summary = summarize(
    trades,
    ["weekday_num", "weekday"],
)

weekday_summary = weekday_summary.sort_values("weekday_num")

print(weekday_summary.to_string(index=False))

weekday_summary.to_csv(
    OUTPUT_DIR / "orb_oos_weekday.csv",
    index=False,
)


# ============================================================
# BREAKOUT TIME
# ============================================================

section("BREAKOUT TIME")

time_summary = summarize(
    trades,
    ["entry_time"],
)

time_summary = time_summary.sort_values("entry_time")

print(time_summary.to_string(index=False))

time_summary.to_csv(
    OUTPUT_DIR / "orb_oos_breakout_time.csv",
    index=False,
)


# ============================================================
# HOLDING TIME
# ============================================================

if "holding_bucket" in trades.columns:
    section("HOLDING TIME")

    holding_summary = summarize(
        trades,
        ["holding_bucket"],
    )

    print(holding_summary.to_string(index=False))

    holding_summary.to_csv(
        OUTPUT_DIR / "orb_oos_holding_time.csv",
        index=False,
    )


# ============================================================
# EXIT REASON
# ============================================================

section("EXIT REASON")

exit_summary = summarize(
    trades,
    ["exit_reason_clean"],
)

print(exit_summary.to_string(index=False))

exit_summary.to_csv(
    OUTPUT_DIR / "orb_oos_exit_reason.csv",
    index=False,
)


# ============================================================
# OR WIDTH
# ============================================================

if "or_width_bucket" in trades.columns:
    section("OPENING RANGE WIDTH")

    width_summary = summarize(
        trades,
        ["or_width_bucket"],
    )

    print(width_summary.to_string(index=False))

    width_summary.to_csv(
        OUTPUT_DIR / "orb_oos_or_width.csv",
        index=False,
    )


# ============================================================
# VOL REGIME
# ============================================================

if "vol_bucket" in trades.columns:
    section("VOL REGIME")

    vol_summary = summarize(
        trades,
        ["vol_bucket"],
    )

    print(vol_summary.to_string(index=False))

    vol_summary.to_csv(
        OUTPUT_DIR / "orb_oos_vol_regime.csv",
        index=False,
    )


# ============================================================
# HMM STATE
# ============================================================

if "hmm_state" in trades.columns:
    section("HMM STATE")

    hmm_summary = summarize(
        trades,
        ["hmm_state"],
    )

    print(hmm_summary.to_string(index=False))

    hmm_summary.to_csv(
        OUTPUT_DIR / "orb_oos_hmm_state.csv",
        index=False,
    )


# ============================================================
# HMM × VOL
# ============================================================

if "hmm_state" in trades.columns and "vol_bucket" in trades.columns:
    section("HMM × VOL")

    hmm_vol = summarize(
        trades,
        [
            "hmm_state",
            "vol_bucket",
        ],
    )

    hmm_vol = hmm_vol.sort_values(
        [
            "hmm_state",
            "vol_bucket",
        ]
    )

    print(hmm_vol.to_string(index=False))

    hmm_vol.to_csv(
        OUTPUT_DIR / "orb_oos_hmm_x_vol.csv",
        index=False,
    )


# ============================================================
# HMM × DIRECTION
# ============================================================

if "hmm_state" in trades.columns:
    section("HMM × DIRECTION")

    hmm_direction = summarize(
        trades,
        [
            "hmm_state",
            "direction",
        ],
    )

    print(hmm_direction.to_string(index=False))

    hmm_direction.to_csv(
        OUTPUT_DIR / "orb_oos_hmm_x_direction.csv",
        index=False,
    )


# ============================================================
# VOL × DIRECTION
# ============================================================

if "vol_bucket" in trades.columns:
    section("VOL × DIRECTION")

    vol_direction = summarize(
        trades,
        [
            "vol_bucket",
            "direction",
        ],
    )

    print(vol_direction.to_string(index=False))

    vol_direction.to_csv(
        OUTPUT_DIR / "orb_oos_vol_x_direction.csv",
        index=False,
    )


# ============================================================
# WEEKDAY × DIRECTION
# ============================================================

section("WEEKDAY × DIRECTION")

weekday_direction = summarize(
    trades,
    [
        "weekday",
        "direction",
    ],
)

print(weekday_direction.to_string(index=False))

weekday_direction.to_csv(
    OUTPUT_DIR / "orb_oos_weekday_x_direction.csv",
    index=False,
)


# ============================================================
# YEARLY OOS STABILITY
# ============================================================

section("YEARLY OOS")

trades["year"] = trades["entry_timestamp"].dt.year

yearly = summarize(
    trades,
    ["year"],
)

print(yearly.to_string(index=False))

yearly.to_csv(
    OUTPUT_DIR / "orb_oos_yearly_context.csv",
    index=False,
)


# ============================================================
# SAVE ENRICHED TRADE FILE
# ============================================================

section("SAVE ENRICHED TRADES")

output_trades = OUTPUT_DIR / "orb_oos_context_enriched.csv"

trades.to_csv(
    output_trades,
    index=False,
)

print(f"Saved: {output_trades}")


# ============================================================
# FINAL AUDIT
# ============================================================

section("FINAL AUDIT")

print(f"Trades analyzed: {len(trades):,}")
print(f"Unique entry timestamps: {trades['entry_timestamp'].nunique():,}")

print(
    f"Date range: "
    f"{trades['entry_timestamp'].min()} "
    f"-> "
    f"{trades['entry_timestamp'].max()}"
)

print()
print("No strategy parameters were optimized.")
print("No trades were removed based on performance.")
print("No regime was selected or excluded.")
print()
print("Context analysis complete.")
