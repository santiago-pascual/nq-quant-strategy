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
    entry               = close of first M5 bar (09:35 ET)
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
import sys

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

ENTRY_OFFSET_BARS = 0  # entry is the closing price of the 09:30-09:35 bar
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

# Make the project root importable when this script is executed directly
# from src\research\mean_reversion\research\.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        df = df.loc[df["market_period"].eq("RTH")].copy()

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
    banner(f"3. BUILD EMA(12), EMA(120), ATR({ATR_PERIOD})")

    x = m5.copy()
    x["ema12"] = x["close"].ewm(span=EMA_SIGNAL, adjust=False).mean()
    x["ema120"] = x["close"].ewm(span=EMA_TRAIL, adjust=False).mean()

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["true_range"] = tr
    # Wilder-style ATR/RMA.
    x["atr"] = tr.ewm(
        alpha=1.0 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD,
    ).mean()

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


def run_strategy(m5: pd.DataFrame) -> pd.DataFrame:
    banner("5. RUN SESSION MOMENTUM BASELINE")

    x = m5.copy().sort_values("timestamp_et").reset_index(drop=True)
    x["session_date"] = x["timestamp_et"].dt.date

    trades: list[dict] = []
    pos: Position | None = None
    sessions_with_entry: set = set()

    # We deliberately process bars chronologically. A position may carry
    # overnight if ALLOW_OVERNIGHT_HOLD=True.
    for i, row in x.iterrows():
        ts = row["timestamp_et"]

        # ------------------------------------------------------------------
        # Manage existing position FIRST.
        # ------------------------------------------------------------------
        if pos is not None:
            # Initial stop remains active until trailing has been activated.
            if pos.side == "LONG":
                stop = (
                    pos.trail_stop
                    if pos.activated and pos.trail_stop is not None
                    else pos.initial_stop
                )
                if row["low"] <= stop:
                    # Conservative OHLC assumption: if stop is touched,
                    # execute at stop. This avoids favorable intrabar pricing.
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

                # Activation is based on the bar's high. The EMA trail is
                # applied only from the NEXT bar, so no close/EMA lookahead.
                activation_level = (
                    pos.entry_price + TRAIL_ACTIVATION_R * pos.risk_points
                )
                if not pos.activated and row["high"] >= activation_level:
                    pos.activated = True
                    pos.activation_time = ts
                    pos.trail_stop = max(pos.initial_stop, float(row["ema120"]))
                elif pos.activated:
                    pos.trail_stop = max(float(pos.trail_stop), float(row["ema120"]))

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
                    pos.trail_stop = min(pos.initial_stop, float(row["ema120"]))
                elif pos.activated:
                    pos.trail_stop = min(float(pos.trail_stop), float(row["ema120"]))

            # Never open another position while one is active.
            continue

        # ------------------------------------------------------------------
        # Entry: first complete M5 bar of the NY session.
        # ------------------------------------------------------------------
        if ts.hour != NY_OPEN_HOUR or ts.minute != NY_OPEN_MINUTE:
            continue

        session = row["session_date"]
        if session in sessions_with_entry:
            continue

        # Entry is at the CLOSE of the opening 09:30-09:35 M5 candle.
        # No second candle is waited for.
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

        entry = float(row["close"])
        risk = float(ATR_MULTIPLIER * row["atr"])

        if risk <= 0 or not np.isfinite(risk):
            continue

        if side == "LONG":
            initial_stop = entry - risk
        else:
            initial_stop = entry + risk

        pos = Position(
            side=side,
            entry_time=ts,
            entry_price=entry,
            risk_points=risk,
            initial_stop=initial_stop,
            signal_time=ts,
            signal_ema12=float(row["ema12"]),
            signal_ema120=float(row["ema120"]),
            signal_atr=float(row["atr"]),
            signal_hmm_state=float(row["hmm_state"])
            if pd.notna(row["hmm_state"])
            else np.nan,
            signal_vol_percentile=float(row["vol_percentile"])
            if pd.notna(row["vol_percentile"])
            else np.nan,
            signal_vol_bucket=str(row["vol_bucket"]),
        )
        sessions_with_entry.add(session)

    # Close any remaining open trade at the last available M5 close.
    if pos is not None:
        last = x.iloc[-1]
        trades.append(
            finalize_trade(
                pos, last["timestamp_et"], float(last["close"]), "END_OF_DATA"
            )
        )

    out = pd.DataFrame(trades)
    if out.empty:
        raise RuntimeError("Session Momentum produced zero trades.")

    out["signal_timestamp"] = pd.to_datetime(out["signal_timestamp"], utc=True)
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"], utc=True)
    out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True)
    out = out.sort_values("entry_timestamp").reset_index(drop=True)

    return out


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
# ATR / OPENING-CANDLE AUDIT
# =============================================================================


def audit_opening_candles(m5: pd.DataFrame) -> pd.DataFrame:
    """Audit that every session has the canonical NY 09:30 opening bar."""
    x = m5.copy()
    x["session_date"] = x["timestamp_et"].dt.date
    opening = x.loc[
        (x["timestamp_et"].dt.hour == 9) & (x["timestamp_et"].dt.minute == 30)
    ].copy()

    counts = (
        x.groupby("session_date")["timestamp_et"]
        .apply(lambda s: int(((s.dt.hour == 9) & (s.dt.minute == 30)).sum()))
        .rename("opening_bar_count")
        .reset_index()
    )

    print("\n" + "=" * 100)
    print("OPENING-CANDLE AUDIT")
    print("=" * 100)
    print("Expected signal candle: 09:30–09:35 America/New_York")
    print("Expected entry candle : 09:35–09:40 America/New_York")
    print(f"Sessions in M5 dataset: {len(counts):,}")
    print(f"Sessions with 09:30 bar: {(counts['opening_bar_count'] == 1).sum():,}")
    print(f"Sessions without 09:30 bar: {(counts['opening_bar_count'] != 1).sum():,}")
    print(f"09:30 bars found: {len(opening):,}")

    if not opening.empty:
        bad_minutes = opening["timestamp_et"].dt.minute.ne(30).sum()
        print(f"Bad opening minute labels: {bad_minutes:,}")

    counts.to_csv(
        RESULTS_ROOT / "parameter_audit" / "opening_candle_audit.csv",
        index=False,
    )
    return counts


def run_atr_period_audit(
    m5_context: pd.DataFrame,
) -> pd.DataFrame:
    """Fast ATR audit.

    Expensive work (Databento load, RTH filtering, M5 construction, HMM/VOL
    context, EMA series and True Range) is performed once. Each ATR period only
    recomputes the rolling Wilder ATR and runs the chronological trade engine.
    """
    global ATR_PERIOD

    audit_dir = RESULTS_ROOT / "parameter_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    periods = list(ATR_AUDIT_PERIODS)
    rows = []

    print("\n" + "=" * 100)
    print("FAST ATR PERIOD AUDIT")
    print("=" * 100)
    print(f"Periods tested: {periods}")
    print("Opening candle: 09:30–09:35 ET")
    print("Entry: CLOSE of opening candle at 09:35 ET")
    print("Context/HMM/VOL is built ONCE and reused for all ATR periods.")

    # Precompute everything independent of ATR period.
    base = m5_context.copy().sort_values("timestamp_et").reset_index(drop=True)
    base["ema12"] = base["close"].ewm(span=EMA_SIGNAL, adjust=False).mean()
    base["ema120"] = base["close"].ewm(span=EMA_TRAIL, adjust=False).mean()
    prev_close = base["close"].shift(1)
    base["true_range"] = pd.concat(
        [
            base["high"] - base["low"],
            (base["high"] - prev_close).abs(),
            (base["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for n, period in enumerate(periods, 1):
        print(f"\n[{n}/{len(periods)}] ATR {period} ...", flush=True)
        ATR_PERIOD = period

        x = base.copy()
        x["atr"] = (
            x["true_range"]
            .ewm(
                alpha=1.0 / period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )

        trades = run_strategy(x)
        trades = trades.loc[
            trades["signal_timestamp"] >= ANALYSIS_START.tz_convert("UTC")
        ].copy()
        trades = trades.sort_values("entry_timestamp").reset_index(drop=True)
        trades["sample"] = np.where(
            trades["signal_timestamp"] <= IS_END.tz_convert("UTC"),
            "IS",
            "OOS",
        )

        full = metrics(trades, "FULL")
        is_m = metrics(trades.loc[trades["sample"].eq("IS")], "IS")
        oos_m = metrics(trades.loc[trades["sample"].eq("OOS")], "OOS")
        oos_trades = trades.loc[trades["sample"].eq("OOS")]

        rows.append(
            {
                "atr_period": period,
                "trades": full.get("trades"),
                "total_R": full.get("total_R"),
                "expectancy_R": full.get("expectancy_R"),
                "win_rate": full.get("win_rate"),
                "profit_factor": full.get("profit_factor"),
                "max_drawdown_R": full.get("max_drawdown_R"),
                "daily_sharpe": full.get("daily_sharpe"),
                "daily_sortino": full.get("daily_sortino"),
                "t_stat": full.get("t_stat"),
                "compound_return_pct": compound_return_pct(trades),
                "compound_dd_pct": max_drawdown_pct_compounded(trades),
                "is_trades": is_m.get("trades"),
                "is_pf": is_m.get("profit_factor"),
                "is_expectancy_R": is_m.get("expectancy_R"),
                "is_win_rate": is_m.get("win_rate"),
                "oos_trades": oos_m.get("trades"),
                "oos_pf": oos_m.get("profit_factor"),
                "oos_expectancy_R": oos_m.get("expectancy_R"),
                "oos_win_rate": oos_m.get("win_rate"),
                "oos_max_drawdown_R": oos_m.get("max_drawdown_R"),
                "oos_sharpe": oos_m.get("daily_sharpe"),
                "oos_t_stat": oos_m.get("t_stat"),
                "oos_compound_return_pct": compound_return_pct(oos_trades),
            }
        )

        print(
            f"[DONE] ATR {period:>2} | "
            f"trades={full.get('trades', 0):>4} | "
            f"PF={full.get('profit_factor', np.nan):.3f} | "
            f"Exp={full.get('expectancy_R', np.nan):+.4f}R | "
            f"OOS PF={oos_m.get('profit_factor', np.nan):.3f} | "
            f"OOS Exp={oos_m.get('expectancy_R', np.nan):+.4f}R",
            flush=True,
        )

    ATR_PERIOD = 14
    result = pd.DataFrame(rows)
    result.to_csv(audit_dir / "atr_period_audit.csv", index=False)

    print("\n" + "=" * 100)
    print("ATR AUDIT — FULL SAMPLE")
    print("=" * 100)
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
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n" + "=" * 100)
    print("ATR AUDIT — OOS")
    print("=" * 100)
    print(
        result[
            [
                "atr_period",
                "oos_trades",
                "oos_pf",
                "oos_expectancy_R",
                "oos_win_rate",
                "oos_max_drawdown_R",
                "oos_sharpe",
                "oos_t_stat",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nSaved: {audit_dir / 'atr_period_audit.csv'}")
    return result


# =============================================================================
# REPORTING
# =============================================================================


def print_metrics_table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    global ATR_PERIOD
    banner("SESSION MOMENTUM — BASELINE RESEARCH")
    print(f"Project root: {ROOT}")
    print(f"Analysis start: {ANALYSIS_START}")
    print(f"IS end:         {IS_END}")
    print(f"ATR period:     {ATR_PERIOD} [PROVISIONAL]")
    print(f"ATR multiplier: {ATR_MULTIPLIER}")
    print(f"EMA signal:     {EMA_SIGNAL}")
    print(f"EMA trail:      {EMA_TRAIL}")
    print(f"Trail trigger:  +{TRAIL_ACTIVATION_R}R")

    rth = load_market()
    m5 = resample_m5(rth)
    events, _ = load_context_sources()

    # Audit opening-candle definition on the full M5 dataset before filtering.
    audit_dir = RESULTS_ROOT / "parameter_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_opening_candles(m5)

    # Build expensive causal context ONCE. The ATR audit reuses this exact
    # context for every period; only the ATR rolling calculation changes.
    ATR_PERIOD = 14
    m5 = add_indicators(m5)
    m5 = build_m5_context(m5, events)

    # Run the ATR reconstruction audit using the exact same strategy engine.
    # Keep the opening candle fixed at 09:30 ET; only ATR period varies.
    m5_audit = (
        m5.loc[m5["timestamp_et"] >= ANALYSIS_START].copy().reset_index(drop=True)
    )
    run_atr_period_audit(m5_audit)

    # Restore baseline ATR for the final baseline report.
    ATR_PERIOD = 14

    # We only use the requested 2020+ sample for the main comparison because
    # the reference screenshot reports 2020-2026.
    m5 = m5.loc[m5["timestamp_et"] >= ANALYSIS_START].copy().reset_index(drop=True)

    trades = run_strategy(m5)

    # Remove trades whose signal occurs before the analysis start. Normally
    # none should remain because the M5 series was filtered first.
    trades = trades.loc[
        trades["signal_timestamp"] >= ANALYSIS_START.tz_convert("UTC")
    ].copy()
    trades = trades.sort_values("entry_timestamp").reset_index(drop=True)

    # IS/OOS based on signal timestamp.
    trades["sample"] = np.where(
        trades["signal_timestamp"] <= IS_END.tz_convert("UTC"),
        "IS",
        "OOS",
    )
    trades["year"] = trades["entry_timestamp"].dt.year
    trades["month"] = (
        trades["entry_timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.to_period("M")
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # A) STRATEGY RESULT ANALYSIS
    # -------------------------------------------------------------------------
    banner("A. STRATEGY RESULT ANALYSIS")

    overall = metrics(trades, "FULL")
    is_trades = trades.loc[trades["sample"].eq("IS")]
    oos_trades = trades.loc[trades["sample"].eq("OOS")]
    is_m = metrics(is_trades, "IS")
    oos_m = metrics(oos_trades, "OOS")

    print("\nFULL")
    print(pd.Series(overall).to_string())
    print("\nIS")
    print(pd.Series(is_m).to_string())
    print("\nOOS")
    print(pd.Series(oos_m).to_string())

    overall_ref_return = compound_return_pct(trades)
    overall_ref_dd = max_drawdown_pct_compounded(trades)
    oos_ref_return = compound_return_pct(oos_trades)
    oos_ref_dd = max_drawdown_pct_compounded(oos_trades)

    reference_comparison = pd.DataFrame(
        [
            [
                "total_trades",
                REFERENCE["total_trades"],
                len(trades),
                len(trades) - REFERENCE["total_trades"],
            ],
            [
                "overall_pf",
                REFERENCE["overall_pf"],
                overall["profit_factor"],
                overall["profit_factor"] - REFERENCE["overall_pf"],
            ],
            [
                "overall_win_rate",
                REFERENCE["overall_win_rate"],
                overall["win_rate"],
                overall["win_rate"] - REFERENCE["overall_win_rate"],
            ],
            [
                "overall_max_dd_pct",
                REFERENCE["overall_max_dd_pct"],
                abs(overall_ref_dd),
                abs(overall_ref_dd) - REFERENCE["overall_max_dd_pct"],
            ],
            [
                "overall_sharpe",
                REFERENCE["overall_sharpe"],
                overall["daily_sharpe"],
                overall["daily_sharpe"] - REFERENCE["overall_sharpe"],
            ],
            [
                "overall_net_return_pct",
                REFERENCE["overall_net_return_pct"],
                overall_ref_return,
                overall_ref_return - REFERENCE["overall_net_return_pct"],
            ],
            [
                "oos_pf",
                REFERENCE["oos_pf"],
                oos_m["profit_factor"],
                oos_m["profit_factor"] - REFERENCE["oos_pf"],
            ],
            [
                "oos_t_stat",
                REFERENCE["oos_t_stat"],
                oos_m["t_stat"],
                oos_m["t_stat"] - REFERENCE["oos_t_stat"],
            ],
            [
                "oos_win_rate",
                REFERENCE["oos_win_rate"],
                oos_m["win_rate"],
                oos_m["win_rate"] - REFERENCE["oos_win_rate"],
            ],
            [
                "oos_expectancy_R",
                REFERENCE["oos_expectancy_r"],
                oos_m["expectancy_R"],
                oos_m["expectancy_R"] - REFERENCE["oos_expectancy_r"],
            ],
            [
                "oos_net_return_pct",
                REFERENCE["oos_net_return_pct"],
                oos_ref_return,
                oos_ref_return - REFERENCE["oos_net_return_pct"],
            ],
        ],
        columns=["metric", "reference_screenshot", "our_reproduction", "difference"],
    )

    print("\nREFERENCE METRIC COMPARISON")
    print(
        reference_comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    by_side = group_metrics(trades, ["side"])
    by_sample_side = group_metrics(trades, ["sample", "side"])
    by_year = group_metrics(trades, ["year"])
    by_month = group_metrics(trades, ["month"])

    # -------------------------------------------------------------------------
    # B) CONTEXT ANALYSIS — HMM / VOL REGIME / SIDE
    # -------------------------------------------------------------------------
    banner("B. HMM + VOLATILITY REGIME ANALYSIS")

    # Ensure context labels are readable and stable.
    trades["hmm_state"] = pd.to_numeric(trades["hmm_state"], errors="coerce").astype(
        "Int64"
    )
    trades["vol_bucket"] = trades["vol_bucket"].astype(str)

    by_hmm = group_metrics(trades, ["hmm_state"])
    by_vol = group_metrics(trades, ["vol_bucket"])
    by_hmm_vol = group_metrics(trades, ["hmm_state", "vol_bucket"])
    by_side_hmm = group_metrics(trades, ["side", "hmm_state"])
    by_side_vol = group_metrics(trades, ["side", "vol_bucket"])
    by_sample_hmm = group_metrics(trades, ["sample", "hmm_state"])
    by_sample_vol = group_metrics(trades, ["sample", "vol_bucket"])

    print("\nBY SIDE")
    print_metrics_table(by_side)
    print("\nBY HMM STATE")
    print_metrics_table(by_hmm)
    print("\nBY VOLATILITY REGIME")
    print_metrics_table(by_vol)
    print("\nBY HMM STATE x VOLATILITY REGIME")
    print_metrics_table(by_hmm_vol)
    print("\nBY SIDE x HMM STATE")
    print_metrics_table(by_side_hmm)
    print("\nBY SIDE x VOLATILITY REGIME")
    print_metrics_table(by_side_vol)

    # -------------------------------------------------------------------------
    # Save strategy results separately from context analysis.
    # -------------------------------------------------------------------------
    trades.to_csv(STRATEGY_DIR / "session_momentum_trades.csv", index=False)
    pd.DataFrame([overall, is_m, oos_m]).to_csv(
        STRATEGY_DIR / "session_momentum_metrics.csv", index=False
    )
    by_side.to_csv(STRATEGY_DIR / "session_momentum_by_side.csv", index=False)
    by_sample_side.to_csv(
        STRATEGY_DIR / "session_momentum_by_sample_side.csv", index=False
    )
    by_year.to_csv(STRATEGY_DIR / "session_momentum_by_year.csv", index=False)
    by_month.to_csv(STRATEGY_DIR / "session_momentum_by_month.csv", index=False)
    reference_comparison.to_csv(
        STRATEGY_DIR / "session_momentum_reference_comparison.csv", index=False
    )

    by_hmm.to_csv(CONTEXT_DIR / "session_momentum_by_hmm_state.csv", index=False)
    by_vol.to_csv(CONTEXT_DIR / "session_momentum_by_vol_regime.csv", index=False)
    by_hmm_vol.to_csv(CONTEXT_DIR / "session_momentum_by_hmm_x_vol.csv", index=False)
    by_side_hmm.to_csv(CONTEXT_DIR / "session_momentum_by_side_x_hmm.csv", index=False)
    by_side_vol.to_csv(CONTEXT_DIR / "session_momentum_by_side_x_vol.csv", index=False)
    by_sample_hmm.to_csv(
        CONTEXT_DIR / "session_momentum_by_sample_x_hmm.csv", index=False
    )
    by_sample_vol.to_csv(
        CONTEXT_DIR / "session_momentum_by_sample_x_vol.csv", index=False
    )

    # A compact configuration file makes the assumptions auditable.
    config = pd.DataFrame(
        [
            ["ema_signal", EMA_SIGNAL, "source screenshot"],
            ["ema_trail", EMA_TRAIL, "source screenshot"],
            ["atr_period", ATR_PERIOD, "PROVISIONAL — screenshot unspecified"],
            ["atr_multiplier", ATR_MULTIPLIER, "source screenshot"],
            ["trail_activation_R", TRAIL_ACTIVATION_R, "source screenshot"],
            [
                "entry",
                "09:35 close of opening M5 candle",
                "user-confirmed implementation",
            ],
            ["atr_formula", "Wilder/RMA", "explicit implementation assumption"],
            ["ema_series", "RTH M5", "explicit implementation assumption"],
            [
                "trail_timing",
                "next bar after activation",
                "explicit no-lookahead assumption",
            ],
            ["session_close", "no forced close", "explicit implementation assumption"],
        ],
        columns=["parameter", "value", "status"],
    )
    config.to_csv(STRATEGY_DIR / "session_momentum_config.csv", index=False)

    banner("FINAL AUDIT")
    print(f"Trades:              {len(trades):,}")
    print(f"LONG:                {(trades['side'] == 'LONG').sum():,}")
    print(f"SHORT:               {(trades['side'] == 'SHORT').sum():,}")
    print(f"Total R:             {trades['r'].sum():+.4f}R")
    print(f"Expectancy:          {trades['r'].mean():+.6f}R")
    print(f"Profit factor:       {profit_factor(trades['r']):.4f}")
    print(f"Win rate:            {(trades['r'] > 0).mean():.4%}")
    print(f"Max DD:              {max_drawdown(trades['r'].cumsum()):+.4f}R")
    print(f"Compounded @ 1.5%:   {overall_ref_return:+.2f}%")
    print(f"Compounded max DD:   {overall_ref_dd:+.2f}%")
    print(f"OOS compounded:      {oos_ref_return:+.2f}%")
    print(f"OOS max DD:          {oos_ref_dd:+.2f}%")
    print(f"\nStrategy results: {STRATEGY_DIR}")
    print(f"Context analysis: {CONTEXT_DIR}")
    print("\nNO ROBUSTNESS TESTING WAS RUN IN THIS SCRIPT.")
    print("Next stage after auditing this baseline: robustness / stress tests.")


if __name__ == "__main__":
    main()
