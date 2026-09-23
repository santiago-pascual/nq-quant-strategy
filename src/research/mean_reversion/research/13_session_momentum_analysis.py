"""
13 — SESSION MOMENTUM: STRATEGY + CONTEXT ANALYSIS

Purpose
-------
Reproduce the "Session Momentum" idea shown in the reference screenshots and
separate two questions:

A) STRATEGY RESULT ANALYSIS
   - trade generation
   - R distribution
   - win rate / PF / expectancy
   - drawdown / Sharpe / Sortino
   - IS vs OOS
   - long vs short
   - comparison with the reference metrics shown in the screenshots

B) MARKET CONTEXT ANALYSIS
   - HMM state at the signal
   - causal volatility regime at the signal
   - HMM x volatility regime
   - long/short x HMM
   - long/short x volatility regime

This script intentionally does NOT perform robustness testing, parameter sweeps,
Monte Carlo, walk-forward optimization, or strategy selection. Those belong in
the next research stage after this baseline has been reproduced and audited.

IMPORTANT SOURCE-INTERPRETATION NOTES
--------------------------------------
The screenshots specify:
    M5
    NY session open
    EMA(12) signal
    EMA(120) trailing stop
    8 ATR initial stop
    activate trailing after +0.5R
    one trade per session
    no fixed target

The screenshots do NOT specify:
    - ATR lookback period
    - exact ATR formula
    - whether EMA/ATR are calculated from RTH-only M5 bars or another series
    - exact entry execution convention
    - exact intrabar behavior of the EMA(120) trail
    - whether a position is forcibly closed at session end

Those choices are therefore explicit configuration below, not hidden assumptions.
Do not call the reproduction "exact" until the source mechanics are confirmed.

Current baseline assumptions:
    ATR period          = 14
    ATR formula         = Wilder/RMA-style EMA of True Range
    M5 data             = RTH 09:30-16:00 ET
    signal              = first 09:30 M5 close vs EMA(12)
    entry               = next M5 bar open (09:35 ET)
    initial stop        = 8 * ATR measured on signal bar
    activation          = intrabar touch of +0.5R
    EMA trail            = EMA(120) known at bar close; applied from NEXT bar
    exit                = initial stop or EMA(120) trailing stop
    session close        = NOT forced; positions may carry to later bars
    one-trade rule       = at most one open position generated per NY session

If the source later gives different mechanics, change the config and rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = ROOT / "src" / "research" / "results" / "session_momentum"
STRATEGY_DIR = RESULTS_ROOT / "strategy_results"
CONTEXT_DIR = RESULTS_ROOT / "context_analysis"

STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPLICIT BASELINE CONFIG
# =============================================================================

EMA_SIGNAL = 12
EMA_TRAIL = 120
ATR_PERIOD = 14  # Baseline only; source screenshot does not specify ATR lookback.
ATR_AUDIT_PERIODS = [5, 7, 10, 12, 14, 16, 20, 24, 30, 40]
ATR_MULTIPLIER = 8.0
TRAIL_ACTIVATION_R = 0.5

NY_OPEN_HOUR = 9
NY_OPEN_MINUTE = 30
RTH_END_HOUR = 16
RTH_END_MINUTE = 0

ENTRY_OFFSET_BARS = 0  # Entry at 09:35 ET close of the 09:30-09:35 opening candle.
ALLOW_OVERNIGHT_HOLD = True

# Reference split shown in the second screenshot.
ANALYSIS_START = pd.Timestamp("2020-01-01", tz="America/New_York")
IS_END = pd.Timestamp("2024-12-31 23:59:59", tz="America/New_York")

# The screenshots report these values. They are comparison targets, not truths
# about our reproduction.
REFERENCE = {
    "total_trades": 1551,
    "overall_pf": 1.29,
    "overall_win_rate": 0.56,
    "overall_max_dd_pct": 25.0,
    "overall_sharpe": 1.54,
    "overall_net_return_pct": 870.0,
    "oos_pf": 1.53,
    "oos_t_stat": 3.0,
    "oos_win_rate": 0.59,
    "oos_expectancy_r": 0.180,
    "oos_net_return_pct": 149.0,
}


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.databento_loader import load_databento_mnq
from src.session_engine import add_session_information


# =============================================================================
# UTILITIES
# =============================================================================


def banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_et(series: pd.Series) -> pd.Series:
    return utc(series).dt.tz_convert("America/New_York")


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity - peak).min())


def profit_factor(r: pd.Series) -> float:
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def streaks(r: pd.Series) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for x in r:
        if x > 0:
            cur_w += 1
            cur_l = 0
            best_w = max(best_w, cur_w)
        elif x < 0:
            cur_l += 1
            cur_w = 0
            best_l = max(best_l, cur_l)
        else:
            cur_w = cur_l = 0
    return best_w, best_l


def sharpe(r: pd.Series) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    # Trade-level annualization is deliberately not used here. For the
    # comparable portfolio framework, daily returns are used instead.
    return float(r.mean() / r.std(ddof=1))


def daily_sharpe(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    daily = trades.groupby(trades["entry_timestamp"].dt.date)["r"].sum()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(252.0) * daily.mean() / daily.std(ddof=1))


def daily_sortino(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    daily = trades.groupby(trades["entry_timestamp"].dt.date)["r"].sum()
    downside = daily[daily < 0]
    if len(downside) == 0:
        return np.inf
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if downside_dev == 0:
        return np.inf
    return float(np.sqrt(252.0) * daily.mean() / downside_dev)


def t_stat(r: pd.Series) -> float:
    r = r.dropna().astype(float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


# =============================================================================
# MARKET DATA
# =============================================================================


def load_market() -> pd.DataFrame:
    banner("1. LOAD CANONICAL DATABENTO DATA")

    df = load_databento_mnq().copy()
    ensure_columns(
        df,
        ["timestamp ET", "open", "high", "low", "close", "volume"],
        "Databento market",
    )

    df["timestamp_et"] = pd.to_datetime(df["timestamp ET"], errors="coerce")
    if df["timestamp_et"].dt.tz is None:
        df["timestamp_et"] = df["timestamp_et"].dt.tz_localize("America/New_York")
    else:
        df["timestamp_et"] = df["timestamp_et"].dt.tz_convert("America/New_York")

    df = df.sort_values("timestamp_et").reset_index(drop=True)
    print(f"1-minute rows: {len(df):,}")
    print(f"Start: {df['timestamp_et'].iloc[0]}")
    print(f"End:   {df['timestamp_et'].iloc[-1]}")

    # Existing project session classification is kept as the canonical source.
    df = add_session_information(df)
    if "market_period" in df.columns:
        df = df.loc[df["market_period"] == "RTH"].copy()

    # Keep only regular NY cash session. This is the interpretation of
    # "NASDAQ session open" used by the strategy screenshot.
    t = df["timestamp_et"].dt.time
    start = pd.Timestamp("09:30").time()
    end = pd.Timestamp("16:00").time()
    df = df.loc[(t >= start) & (t < end)].copy()

    return df


# =============================================================================
# M5 CONSTRUCTION + INDICATORS
# =============================================================================


def resample_m5(rth: pd.DataFrame) -> pd.DataFrame:
    banner("2. BUILD RTH 5-MINUTE BARS")

    x = rth.copy().set_index("timestamp_et")

    # The source is 1-minute data. Since RTH starts exactly at 09:30 ET,
    # flooring each timestamp to 5 minutes gives 09:30, 09:35, ... bars.
    x["bar"] = x.index.floor("5min")

    m5 = (
        x.groupby("bar", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            minute_count=("close", "count"),
        )
        .reset_index()
        .rename(columns={"bar": "timestamp_et"})
    )

    # Do not use incomplete 5-minute bars.
    m5 = m5.loc[m5["minute_count"] == 5].copy()
    m5["session_date"] = m5["timestamp_et"].dt.date
    m5 = m5.sort_values("timestamp_et").reset_index(drop=True)

    print(f"Complete RTH M5 bars: {len(m5):,}")
    print(f"Sessions: {m5['session_date'].nunique():,}")

    return m5


def add_indicators(m5: pd.DataFrame) -> pd.DataFrame:
    banner("3. BUILD EMA(12), EMA(120), ATR AUDIT GRID")

    x = m5.copy()
    x["ema12"] = x["close"].ewm(span=EMA_SIGNAL, adjust=False).mean()
    x["ema120"] = x["close"].ewm(span=EMA_TRAIL, adjust=False).mean()

    prev_close = x["close"].shift(1)
    x["true_range"] = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Compute every ATR once. The HMM/VOL context is independent of ATR and
    # is therefore NOT recomputed for every period.
    for period in ATR_AUDIT_PERIODS:
        x[f"atr_{period}"] = (
            x["true_range"]
            .ewm(
                alpha=1.0 / period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )

    # Baseline column consumed by run_strategy().
    x["atr"] = x[f"atr_{ATR_PERIOD}"]

    return x


# =============================================================================
# HMM + VOLATILITY CONTEXT
# =============================================================================


def load_context_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the same canonical HMM/event context used by 08AA."""

    cache = ROOT / "src" / "research" / "mean_reversion" / "results" / "cache"
    events_path = cache / "research_07_event_metadata.csv"
    hmm_path = cache / "research_08b_causal_hmm_states.csv"

    if not events_path.exists() or not hmm_path.exists():
        raise FileNotFoundError(
            "Canonical Research 07/08B cache not found. Expected:\n"
            f"{events_path}\n{hmm_path}"
        )

    events = pd.read_csv(events_path)
    hmm = pd.read_csv(hmm_path)

    ensure_columns(
        events,
        ["event_id", "timestamp", "close", "zscore_30"],
        "Research 07 event metadata",
    )
    ensure_columns(
        hmm,
        ["event_id", "timestamp", "hmm_state"],
        "Research 08B HMM",
    )

    events["timestamp"] = utc(events["timestamp"])
    hmm["timestamp"] = utc(hmm["timestamp"])
    events["hmm_state"] = (
        hmm.set_index("event_id")["hmm_state"].reindex(events["event_id"]).to_numpy()
    )
    events["timestamp_et"] = events["timestamp"].dt.tz_convert("America/New_York")

    return events, hmm


def causal_percentile(values: pd.Series) -> pd.Series:
    """Expanding percentile using only observations strictly before t."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(arr), np.nan)
    history: list[float] = []

    for i, v in enumerate(arr):
        if np.isfinite(v) and history:
            a = np.sort(np.asarray(history, dtype=float))
            out[i] = np.searchsorted(a, v, side="right") / len(a)
        if np.isfinite(v):
            history.append(float(v))

    return pd.Series(out, index=values.index)


def build_m5_context(m5: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    banner("4. MAP CAUSAL HMM + VOLATILITY CONTEXT TO M5 SIGNAL BARS")

    # We use the existing canonical 1-minute realized_vol_30 path rather than
    # inventing a second volatility definition for regime classification.
    market = load_databento_mnq().copy()
    market = add_session_information(market)

    ts = pd.to_datetime(market["timestamp ET"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    market["timestamp_et"] = ts

    # The canonical project feature pipeline is imported here to keep regime
    # labels identical to the mean-reversion research.
    from src.feature_engine import add_return_features, add_volatility_features

    market = add_return_features(market)
    market = add_volatility_features(market)
    market = market.loc[market["market_period"].eq("RTH")].copy()
    market = market.sort_values("timestamp_et")

    market["vol_percentile"] = causal_percentile(market["realized_vol_30"])
    p = market["vol_percentile"]
    market["vol_bucket"] = "UNKNOWN"
    market.loc[p < 0.20, "vol_bucket"] = "VOL0-20"
    market.loc[(p >= 0.20) & (p < 0.40), "vol_bucket"] = "VOL20-40"
    market.loc[(p >= 0.40) & (p < 0.60), "vol_bucket"] = "VOL40-60"
    market.loc[(p >= 0.60) & (p < 0.80), "vol_bucket"] = "VOL60-80"
    market.loc[p >= 0.80, "vol_bucket"] = "VOL80-100"

    # For every M5 timestamp, use the canonical 1-minute state/regime observed
    # at the exact signal-bar timestamp (09:30). This is an entry-context label,
    # not a strategy filter.
    event = events[["timestamp", "timestamp_et", "hmm_state"]].drop_duplicates(
        "timestamp"
    )

    lookup = market[
        ["timestamp_et", "realized_vol_30", "vol_percentile", "vol_bucket"]
    ].copy()

    lookup["timestamp"] = lookup["timestamp_et"].dt.tz_convert("UTC")

    event = event.merge(lookup, on="timestamp", how="left")
    event = event.rename(columns={"timestamp_et_x": "event_timestamp_et"})

    out = m5.copy()
    out = out.merge(
        event[
            [
                "timestamp",
                "hmm_state",
                "realized_vol_30",
                "vol_percentile",
                "vol_bucket",
            ]
        ].rename(columns={"timestamp": "signal_timestamp_utc"}),
        left_on="timestamp_et",
        right_on="signal_timestamp_utc",
        how="left",
    )

    return out.drop(columns=["signal_timestamp_utc"], errors="ignore")


# =============================================================================
# STRATEGY ENGINE
# =============================================================================


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    risk_points: float
    initial_stop: float
    signal_time: pd.Timestamp
    signal_ema12: float
    signal_ema120: float
    signal_atr: float
    signal_hmm_state: float
    signal_vol_percentile: float
    signal_vol_bucket: str
    activated: bool = False
    trail_stop: float | None = None
    activation_time: pd.Timestamp | None = None


def finalize_trade(
    pos: Position,
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: str,
) -> dict:
    if pos.side == "LONG":
        r = (exit_price - pos.entry_price) / pos.risk_points
    else:
        r = (pos.entry_price - exit_price) / pos.risk_points

    return {
        "strategy": "SessionMomentum",
        "side": pos.side,
        "signal_timestamp": pos.signal_time,
        "entry_timestamp": pos.entry_time,
        "exit_timestamp": exit_time,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "risk_points": pos.risk_points,
        "initial_stop": pos.initial_stop,
        "signal_ema12": pos.signal_ema12,
        "signal_ema120": pos.signal_ema120,
        "signal_atr": pos.signal_atr,
        "trail_activated": pos.activated,
        "activation_timestamp": pos.activation_time,
        "exit_reason": reason,
        "r": r,
        "hmm_state": pos.signal_hmm_state,
        "vol_percentile": pos.signal_vol_percentile,
        "vol_bucket": pos.signal_vol_bucket,
    }


def run_strategy(
    m5: pd.DataFrame, atr_period: int | None = None, verbose: bool = True
) -> pd.DataFrame:
    if verbose:
        banner(f"5. RUN SESSION MOMENTUM — ATR({atr_period or ATR_PERIOD})")

    x = m5.copy().sort_values("timestamp_et").reset_index(drop=True)
    x["session_date"] = x["timestamp_et"].dt.date

    if atr_period is None:
        atr_period = ATR_PERIOD

    atr_col = f"atr_{atr_period}"
    if atr_col in x.columns:
        x["atr"] = x[atr_col]
    elif "atr" not in x.columns:
        raise KeyError(f"Missing ATR column: {atr_col}")

    trades: list[dict] = []
    pos: Position | None = None
    sessions_with_entry: set = set()

    for i, row in x.iterrows():
        ts = row["timestamp_et"]

        # Manage existing position first.
        if pos is not None:
            if pos.side == "LONG":
                stop = (
                    pos.trail_stop
                    if pos.activated and pos.trail_stop is not None
                    else pos.initial_stop
                )

                if row["low"] <= stop:
                    trades.append(
                        finalize_trade(
                            pos,
                            ts,
                            stop,
                            "TRAIL_STOP" if pos.activated else "INITIAL_STOP",
                        )
                    )
                    pos = None
                    continue

                activation_level = (
                    pos.entry_price + TRAIL_ACTIVATION_R * pos.risk_points
                )

                if not pos.activated and row["high"] >= activation_level:
                    pos.activated = True
                    pos.activation_time = ts
                    pos.trail_stop = max(
                        pos.initial_stop,
                        float(row["ema120"]),
                    )
                elif pos.activated:
                    pos.trail_stop = max(
                        float(pos.trail_stop),
                        float(row["ema120"]),
                    )

            else:
                stop = (
                    pos.trail_stop
                    if pos.activated and pos.trail_stop is not None
                    else pos.initial_stop
                )

                if row["high"] >= stop:
                    trades.append(
                        finalize_trade(
                            pos,
                            ts,
                            stop,
                            "TRAIL_STOP" if pos.activated else "INITIAL_STOP",
                        )
                    )
                    pos = None
                    continue

                activation_level = (
                    pos.entry_price - TRAIL_ACTIVATION_R * pos.risk_points
                )

                if not pos.activated and row["low"] <= activation_level:
                    pos.activated = True
                    pos.activation_time = ts
                    pos.trail_stop = min(
                        pos.initial_stop,
                        float(row["ema120"]),
                    )
                elif pos.activated:
                    pos.trail_stop = min(
                        float(pos.trail_stop),
                        float(row["ema120"]),
                    )

            continue

        # --------------------------------------------------------------
        # First NY opening candle: 09:30 -> 09:35 ET.
        # Direction is determined by its CLOSE vs EMA12.
        # ENTRY is exactly at 09:35 ET using that candle's CLOSE.
        # --------------------------------------------------------------
        if ts.hour != NY_OPEN_HOUR or ts.minute != NY_OPEN_MINUTE:
            continue

        session = row["session_date"]

        if session in sessions_with_entry:
            continue

        if not np.isfinite(row["ema12"]) or not np.isfinite(row["ema120"]):
            continue

        if not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue

        if row["close"] > row["ema12"]:
            side = "LONG"
        elif row["close"] < row["ema12"]:
            side = "SHORT"
        else:
            continue

        # IMPORTANT: close of the opening candle, not next bar open.
        entry = float(row["close"])
        entry_time = ts + pd.Timedelta(minutes=5)

        risk = float(ATR_MULTIPLIER * row["atr"])

        if risk <= 0 or not np.isfinite(risk):
            continue

        initial_stop = entry - risk if side == "LONG" else entry + risk

        pos = Position(
            side=side,
            entry_time=entry_time,
            entry_price=entry,
            risk_points=risk,
            initial_stop=initial_stop,
            signal_time=ts,
            signal_ema12=float(row["ema12"]),
            signal_ema120=float(row["ema120"]),
            signal_atr=float(row["atr"]),
            signal_hmm_state=(
                float(row["hmm_state"]) if pd.notna(row["hmm_state"]) else np.nan
            ),
            signal_vol_percentile=(
                float(row["vol_percentile"])
                if pd.notna(row["vol_percentile"])
                else np.nan
            ),
            signal_vol_bucket=str(row["vol_bucket"]),
        )

        sessions_with_entry.add(session)

    if pos is not None:
        last = x.iloc[-1]
        trades.append(
            finalize_trade(
                pos,
                last["timestamp_et"],
                float(last["close"]),
                "END_OF_DATA",
            )
        )

    out = pd.DataFrame(trades)

    if out.empty:
        raise RuntimeError("Session Momentum produced zero trades.")

    out["signal_timestamp"] = pd.to_datetime(out["signal_timestamp"], utc=True)
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"], utc=True)
    out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True)

    return out.sort_values("entry_timestamp").reset_index(drop=True)


# =============================================================================
# METRICS
# =============================================================================


def metrics(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {
            "sample": label,
            "trades": 0,
        }

    r = trades["r"].astype(float)
    eq = r.cumsum()
    best_w, best_l = streaks(r)

    wins = r[r > 0]
    losses = r[r < 0]

    return {
        "sample": label,
        "trades": len(r),
        "total_R": r.sum(),
        "expectancy_R": r.mean(),
        "median_R": r.median(),
        "win_rate": (r > 0).mean(),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": max_drawdown(eq),
        "average_win_R": wins.mean() if not wins.empty else np.nan,
        "average_loss_R": losses.mean() if not losses.empty else np.nan,
        "payoff_ratio": wins.mean() / abs(losses.mean())
        if not wins.empty and not losses.empty
        else np.nan,
        "t_stat": t_stat(r),
        "daily_sharpe": daily_sharpe(trades),
        "daily_sortino": daily_sortino(trades),
        "longest_win_streak": best_w,
        "longest_loss_streak": best_l,
        "activated_pct": trades["trail_activated"].mean(),
    }


def group_metrics(
    trades: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, g in trades.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metrics(g, "group"))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_reference(actual: dict) -> pd.DataFrame:
    mapping = {
        "total_trades": "trades",
        "overall_pf": "profit_factor",
        "overall_win_rate": "win_rate",
        "overall_sharpe": "daily_sharpe",
        "oos_pf": "oos_profit_factor",
        "oos_win_rate": "oos_win_rate",
        "oos_expectancy_r": "oos_expectancy_R",
    }

    rows = []
    for reference_name, actual_name in mapping.items():
        if actual_name in actual:
            ref = REFERENCE[reference_name]
            val = actual[actual_name]
            rows.append(
                {
                    "metric": reference_name,
                    "reference": ref,
                    "reproduction": val,
                    "difference": val - ref if np.isfinite(val) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# COMPOUNDED RETURN AT 1.5% RISK — REFERENCE METRIC ONLY
# =============================================================================


def compound_return_pct(trades: pd.DataFrame, risk_pct: float = 0.015) -> float:
    """Compound 1.5% of current equity per trade using the trade R result."""
    if trades.empty:
        return np.nan
    equity = 1.0
    for r in trades["r"]:
        equity *= 1.0 + risk_pct * float(r)
    return (equity - 1.0) * 100.0


def max_drawdown_pct_compounded(trades: pd.DataFrame, risk_pct: float = 0.015) -> float:
    if trades.empty:
        return np.nan
    equity = 1.0
    peak = 1.0
    dd = 0.0
    for r in trades["r"]:
        equity *= 1.0 + risk_pct * float(r)
        peak = max(peak, equity)
        dd = min(dd, equity / peak - 1.0)
    return dd * 100.0


# =============================================================================
# REPORTING
# =============================================================================


def print_metrics_table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def opening_candle_audit(m5: pd.DataFrame) -> pd.DataFrame:
    banner("OPENING-CANDLE AUDIT")

    sessions = m5["session_date"].nunique()
    opening = m5.loc[
        (m5["timestamp_et"].dt.hour == NY_OPEN_HOUR)
        & (m5["timestamp_et"].dt.minute == NY_OPEN_MINUTE)
    ].copy()

    bad = opening.loc[~opening["timestamp_et"].dt.minute.eq(NY_OPEN_MINUTE)]

    print("Expected signal candle: 09:30–09:35 America/New_York")
    print("Expected entry time   : 09:35 America/New_York")
    print("Expected entry price  : CLOSE of 09:30–09:35 opening candle")
    print(f"Sessions in M5 dataset: {sessions:,}")
    print(f"Sessions with 09:30 bar: {opening['session_date'].nunique():,}")
    print(
        f"Sessions without 09:30 bar: {sessions - opening['session_date'].nunique():,}"
    )
    print(f"09:30 bars found: {len(opening):,}")
    print(f"Bad opening minute labels: {len(bad):,}")

    audit = pd.DataFrame(
        {
            "metric": [
                "sessions",
                "sessions_with_0930",
                "sessions_without_0930",
                "opening_bars",
                "bad_opening_minute_labels",
            ],
            "value": [
                sessions,
                opening["session_date"].nunique(),
                sessions - opening["session_date"].nunique(),
                len(opening),
                len(bad),
            ],
        }
    )
    return audit


def run_atr_audit(m5: pd.DataFrame) -> pd.DataFrame:
    banner("ATR PERIOD AUDIT")
    print(f"Periods tested: {ATR_AUDIT_PERIODS}")
    print(
        "Opening candle fixed at 09:30–09:35 ET; "
        "entry = 09:35 ET close of opening candle."
    )
    print(
        "HMM/VOL context and EMA12/EMA120 are computed ONCE and reused "
        "across all ATR periods."
    )

    rows = []

    for period in ATR_AUDIT_PERIODS:
        print(f"\nRunning ATR({period}) ...", flush=True)

        trades = run_strategy(
            m5,
            atr_period=period,
            verbose=False,
        )

        trades = trades.loc[
            trades["signal_timestamp"] >= ANALYSIS_START.tz_convert("UTC")
        ].copy()

        trades["sample"] = np.where(
            trades["signal_timestamp"] <= IS_END.tz_convert("UTC"),
            "IS",
            "OOS",
        )

        full = metrics(trades, "FULL")
        is_m = metrics(
            trades.loc[trades["sample"].eq("IS")],
            "IS",
        )
        oos_m = metrics(
            trades.loc[trades["sample"].eq("OOS")],
            "OOS",
        )

        rows.append(
            {
                "atr_period": period,
                "trades": full["trades"],
                "total_R": full["total_R"],
                "expectancy_R": full["expectancy_R"],
                "win_rate": full["win_rate"],
                "profit_factor": full["profit_factor"],
                "max_drawdown_R": full["max_drawdown_R"],
                "daily_sharpe": full["daily_sharpe"],
                "daily_sortino": full["daily_sortino"],
                "t_stat": full["t_stat"],
                "is_trades": is_m["trades"],
                "is_total_R": is_m["total_R"],
                "is_expectancy_R": is_m["expectancy_R"],
                "is_win_rate": is_m["win_rate"],
                "is_profit_factor": is_m["profit_factor"],
                "oos_trades": oos_m["trades"],
                "oos_total_R": oos_m["total_R"],
                "oos_expectancy_R": oos_m["expectancy_R"],
                "oos_win_rate": oos_m["win_rate"],
                "oos_profit_factor": oos_m["profit_factor"],
                "oos_max_drawdown_R": oos_m["max_drawdown_R"],
                "oos_daily_sharpe": oos_m["daily_sharpe"],
                "oos_t_stat": oos_m["t_stat"],
            }
        )

    result = pd.DataFrame(rows)

    print("\n" + "=" * 140)
    print("ATR AUDIT — FULL SAMPLE")
    print("=" * 140)
    print(
        result[
            [
                "atr_period",
                "trades",
                "total_R",
                "expectancy_R",
                "win_rate",
                "profit_factor",
                "max_drawdown_R",
                "daily_sharpe",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 140)
    print("ATR AUDIT — OOS")
    print("=" * 140)
    print(
        result[
            [
                "atr_period",
                "oos_trades",
                "oos_total_R",
                "oos_expectancy_R",
                "oos_win_rate",
                "oos_profit_factor",
                "oos_max_drawdown_R",
                "oos_daily_sharpe",
                "oos_t_stat",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    outdir = RESULTS_ROOT / "parameter_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        outdir / "atr_period_audit.csv",
        index=False,
    )

    print(f"\nSaved ATR audit: {outdir / 'atr_period_audit.csv'}")

    return result


def main() -> None:
    banner("SESSION MOMENTUM — ATR / OPENING-CANDLE AUDIT")

    print(f"Project root: {ROOT}")
    print(f"Analysis start: {ANALYSIS_START}")
    print(f"IS end:         {IS_END}")
    print(f"ATR periods:    {ATR_AUDIT_PERIODS}")
    print(f"ATR multiplier: {ATR_MULTIPLIER}")
    print(f"EMA signal:     {EMA_SIGNAL}")
    print(f"EMA trail:      {EMA_TRAIL}")
    print(f"Trail trigger:  +{TRAIL_ACTIVATION_R}R")
    print("Opening candle: 09:30–09:35 ET")
    print("Entry:          09:35 ET close of opening candle")

    # ==============================================================
    # EXPENSIVE STEPS — EACH RUNS ONLY ONCE
    # ==============================================================

    rth = load_market()
    m5 = resample_m5(rth)

    opening_audit = opening_candle_audit(m5)

    # Compute EMA / TR / ALL ATR PERIODS once.
    m5 = add_indicators(m5)

    # Compute HMM/VOL context once.
    events, _ = load_context_sources()
    m5 = build_m5_context(m5, events)

    m5 = m5.loc[m5["timestamp_et"] >= ANALYSIS_START].copy().reset_index(drop=True)

    # ==============================================================
    # CHEAP STEP — 10 STRATEGY RUNS
    # ==============================================================

    atr_results = run_atr_audit(m5)

    # ==============================================================
    # BASELINE = ATR(14)
    # ==============================================================

    baseline = atr_results.loc[atr_results["atr_period"].eq(ATR_PERIOD)]

    if baseline.empty:
        raise RuntimeError("Baseline ATR(14) result missing.")

    print("\n" + "=" * 100)
    print("BASELINE ATR(14)")
    print("=" * 100)
    print(
        baseline.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ==============================================================
    # OPENING CANDLE AUDIT OUTPUT
    # ==============================================================

    outdir = RESULTS_ROOT / "parameter_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    opening_audit.to_csv(
        outdir / "opening_candle_audit.csv",
        index=False,
    )

    banner("FINAL AUDIT")
    print("Opening candle: 09:30–09:35 ET")
    print("Entry:          09:35 ET CLOSE")
    print(f"ATR periods:    {ATR_AUDIT_PERIODS}")
    print("HMM/VOL context: computed once and reused for every ATR period.")
    print("No robustness, Monte Carlo, or automatic parameter selection was performed.")
    print(f"\nResults: {outdir}")


if __name__ == "__main__":
    main()
