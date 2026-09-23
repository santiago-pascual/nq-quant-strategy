"""
SESSION MOMENTUM — EXECUTION / EMA120 TRAILING AUDIT

Purpose
-------
Audit the unresolved execution mechanics of the Session Momentum strategy.

FIXED STRATEGY
--------------
- NASDAQ / MNQ
- M5
- NY RTH
- Opening candle: 09:30–09:35 ET
- Entry: 09:35 ET CLOSE
- Direction:
    LONG  if opening close > EMA12
    SHORT if opening close < EMA12
- Initial stop: 8 * ATR14
- Trail activates after +0.5R
- EMA120 trailing
- One trade per session
- No fixed TP
- Position may carry beyond RTH

ONLY THESE MECHANICS VARY
-------------------------
A) Activation based on intrabar High/Low vs Close
B) EMA120 exit based on intrabar touch vs Close
C) EMA120 evaluated from current bar vs previous completed bar
D) Activation bar allowed to execute EMA120 exit vs next bar
E) Combination of the above

IMPORTANT
---------
This is an execution-mechanics audit.
It does NOT change:
- ATR period
- ATR multiplier
- EMA12
- entry
- one-trade-per-session rule
- signal logic
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIG
# ============================================================================

ATR_PERIOD = 14
ATR_MULTIPLIER = 8.0

EMA_SIGNAL = 12
EMA_TRAIL = 120

TRAIL_TRIGGER_R = 0.5

ENTRY_HOUR = 9
ENTRY_MINUTE = 35

SESSION_START_HOUR = 9
SESSION_START_MINUTE = 30

SESSION_END_HOUR = 16
SESSION_END_MINUTE = 0

INITIAL_CAPITAL_R = 0.0


# ============================================================================
# IMPORT CANONICAL LOADER
# ============================================================================

try:
    from src.databento_loader import load_databento_mnq
except Exception:
    from src.data_loader import load_data as load_databento_mnq


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Trade:
    strategy: str
    side: str

    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp

    entry_price: float
    exit_price: float

    stop_price: float

    initial_r_points: float

    pnl_points: float
    net_R: float

    bars_held: int

    activated: bool
    activation_timestamp: Optional[pd.Timestamp]

    exit_reason: str

    ema_exit_value: Optional[float]


# ============================================================================
# HELPERS
# ============================================================================


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the canonical Databento dataframe into:
        timestamp ET
        open
        high
        low
        close
        volume
    """

    out = df.copy()

    rename = {}

    for c in out.columns:
        lc = str(c).lower().strip()

        if lc in {"timestamp et", "timestamp_et", "timestamp"}:
            rename[c] = "timestamp ET"
        elif lc == "open":
            rename[c] = "open"
        elif lc == "high":
            rename[c] = "high"
        elif lc == "low":
            rename[c] = "low"
        elif lc == "close":
            rename[c] = "close"
        elif lc == "volume":
            rename[c] = "volume"

    out = out.rename(columns=rename)

    required = [
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [x for x in required if x not in out.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(out.columns)}"
        )

    ts = pd.to_datetime(out["timestamp ET"])

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")

    out["timestamp ET"] = ts.dt.tz_convert("America/New_York")

    out = out.sort_values("timestamp ET").reset_index(drop=True)

    return out


def build_m5(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete 5-minute RTH candles.
    """

    df = df_1m.copy()

    ts = df["timestamp ET"]

    df = df[
        (ts.dt.hour > SESSION_START_HOUR)
        | ((ts.dt.hour == SESSION_START_HOUR) & (ts.dt.minute >= SESSION_START_MINUTE))
    ].copy()

    df = df[
        (df["timestamp ET"].dt.hour < SESSION_END_HOUR)
        | (
            (df["timestamp ET"].dt.hour == SESSION_END_HOUR)
            & (df["timestamp ET"].dt.minute < SESSION_END_MINUTE)
        )
    ].copy()

    df["bar_time"] = df["timestamp ET"].dt.floor("5min")

    m5 = (
        df.groupby("bar_time", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum") if "volume" in df.columns else ("close", "size"),
            n_1m=("close", "size"),
        )
        .reset_index()
        .rename(columns={"bar_time": "timestamp ET"})
    )

    # Complete M5 candle = 5 one-minute observations.
    m5 = m5[m5["n_1m"] == 5].copy()

    m5["session_date"] = m5["timestamp ET"].dt.date

    return m5.reset_index(drop=True)


def add_indicators(m5: pd.DataFrame) -> pd.DataFrame:
    df = m5.copy()

    df["ema12"] = df["close"].ewm(span=EMA_SIGNAL, adjust=False).mean()

    df["ema120"] = df["close"].ewm(span=EMA_TRAIL, adjust=False).mean()

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder / RMA ATR
    df["atr14"] = df["tr"].ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

    return df


# ============================================================================
# METRICS
# ============================================================================


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()

    if losses == 0:
        return np.nan

    return float(gains / losses)


def max_drawdown(values: pd.Series) -> float:
    equity = values.cumsum()

    peak = equity.cummax()

    dd = equity - peak

    return float(dd.min())


def daily_sharpe(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan

    daily = trades.groupby(trades["exit_timestamp"].dt.date)["net_R"].sum()

    if len(daily) < 2:
        return np.nan

    std = daily.std(ddof=1)

    if std == 0:
        return np.nan

    return float(daily.mean() / std * math.sqrt(252))


def summarize(trades: pd.DataFrame, label: str) -> dict:

    if trades.empty:
        return {
            "variant": label,
            "trades": 0,
            "total_R": np.nan,
            "expectancy_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "daily_sharpe": np.nan,
        }

    r = trades["net_R"]

    return {
        "variant": label,
        "trades": len(trades),
        "total_R": r.sum(),
        "expectancy_R": r.mean(),
        "win_rate": (r > 0).mean(),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": max_drawdown(r),
        "daily_sharpe": daily_sharpe(trades),
    }


# ============================================================================
# STRATEGY ENGINE
# ============================================================================


def run_strategy(
    df: pd.DataFrame,
    activation_mode: str,
    ema_exit_mode: str,
    ema_source: str,
    activation_bar_trail: bool,
) -> pd.DataFrame:

    trades = []

    grouped = df.groupby("session_date", sort=True)

    for session_date, session in grouped:
        session = session.sort_values("timestamp ET").reset_index(drop=True)

        # ------------------------------------------------------------------
        # OPENING CANDLE
        # ------------------------------------------------------------------

        opening = session[
            (session["timestamp ET"].dt.hour == 9)
            & (session["timestamp ET"].dt.minute == 30)
        ]

        if opening.empty:
            continue

        opening = opening.iloc[0]

        # Signal
        if opening["close"] > opening["ema12"]:
            side = "LONG"

        elif opening["close"] < opening["ema12"]:
            side = "SHORT"

        else:
            continue

        entry_time = opening["timestamp ET"] + pd.Timedelta(minutes=5)

        entry_rows = session[session["timestamp ET"] == entry_time]

        if entry_rows.empty:
            continue

        entry_bar_idx = entry_rows.index[0]

        entry_price = float(opening["close"])

        atr = float(opening["atr14"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        risk_points = ATR_MULTIPLIER * atr

        if side == "LONG":
            initial_stop = entry_price - risk_points
        else:
            initial_stop = entry_price + risk_points

        activated = False
        activation_timestamp = None

        exit_price = None
        exit_timestamp = None
        exit_reason = None
        ema_exit_value = None

        # ------------------------------------------------------------------
        # TRADE LOOP
        # ------------------------------------------------------------------

        trade_slice = session.iloc[entry_bar_idx:].copy()

        for i in range(len(trade_slice)):
            row = trade_slice.iloc[i]

            ts = row["timestamp ET"]

            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            # ==============================================================
            # 1. INITIAL STOP
            # ==============================================================

            if side == "LONG":
                stop_hit = low <= initial_stop

            else:
                stop_hit = high >= initial_stop

            if stop_hit:
                exit_price = initial_stop
                exit_timestamp = ts
                exit_reason = "INITIAL_STOP"

                break

            # ==============================================================
            # 2. DETECT +0.5R ACTIVATION
            # ==============================================================

            if not activated:
                trigger_distance = TRAIL_TRIGGER_R * risk_points

                if activation_mode == "INTRABAR":
                    if side == "LONG":
                        activated = high >= (entry_price + trigger_distance)
                    else:
                        activated = low <= (entry_price - trigger_distance)

                elif activation_mode == "CLOSE":
                    if side == "LONG":
                        activated = close >= (entry_price + trigger_distance)
                    else:
                        activated = close <= (entry_price - trigger_distance)

                if activated:
                    activation_timestamp = ts

            # ==============================================================
            # 3. EMA120 TRAILING
            # ==============================================================

            if activated:
                if not activation_bar_trail and ts == activation_timestamp:
                    continue

                # ----------------------------------------------------------
                # EMA source
                # ----------------------------------------------------------

                if ema_source == "CURRENT":
                    ema = float(row["ema120"])

                elif ema_source == "PREVIOUS":
                    if i == 0:
                        continue

                    ema = float(trade_slice.iloc[i - 1]["ema120"])

                else:
                    raise ValueError(f"Unknown ema_source={ema_source}")

                if not np.isfinite(ema):
                    continue

                # ----------------------------------------------------------
                # LONG
                # ----------------------------------------------------------

                if side == "LONG":
                    if ema_exit_mode == "INTRABAR":
                        hit = low <= ema

                    elif ema_exit_mode == "CLOSE":
                        hit = close <= ema

                    else:
                        raise ValueError(f"Unknown ema_exit_mode={ema_exit_mode}")

                    if hit:
                        exit_price = ema
                        exit_timestamp = ts
                        exit_reason = "EMA120_TRAIL"
                        ema_exit_value = ema

                        break

                # ----------------------------------------------------------
                # SHORT
                # ----------------------------------------------------------

                else:
                    if ema_exit_mode == "INTRABAR":
                        hit = high >= ema

                    elif ema_exit_mode == "CLOSE":
                        hit = close >= ema

                    else:
                        raise ValueError(f"Unknown ema_exit_mode={ema_exit_mode}")

                    if hit:
                        exit_price = ema
                        exit_timestamp = ts
                        exit_reason = "EMA120_TRAIL"
                        ema_exit_value = ema

                        break

        # ------------------------------------------------------------------
        # END OF DATA
        # ------------------------------------------------------------------

        if exit_price is None:
            last = trade_slice.iloc[-1]

            exit_price = float(last["close"])
            exit_timestamp = last["timestamp ET"]
            exit_reason = "END_OF_DATA"

        # ------------------------------------------------------------------
        # PNL
        # ------------------------------------------------------------------

        if side == "LONG":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        net_R = pnl_points / risk_points

        bars_held = len(
            session[
                (session["timestamp ET"] >= entry_time)
                & (session["timestamp ET"] <= exit_timestamp)
            ]
        )

        trades.append(
            {
                "strategy": "SESSION_MOMENTUM",
                "side": side,
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": initial_stop,
                "initial_r_points": risk_points,
                "pnl_points": pnl_points,
                "net_R": net_R,
                "bars_held": bars_held,
                "activated": activated,
                "activation_timestamp": activation_timestamp,
                "exit_reason": exit_reason,
                "ema_exit_value": ema_exit_value,
            }
        )

    return pd.DataFrame(trades)


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 100)
    print("SESSION MOMENTUM — EXECUTION / EMA120 TRAILING AUDIT")
    print("=" * 100)

    print()
    print(f"ATR period:       {ATR_PERIOD}")
    print(f"ATR multiplier:   {ATR_MULTIPLIER}")
    print(f"EMA signal:       {EMA_SIGNAL}")
    print(f"EMA trail:        {EMA_TRAIL}")
    print(f"Trail trigger:    +{TRAIL_TRIGGER_R}R")
    print("Opening candle:   09:30–09:35 ET")
    print("Entry:            09:35 ET CLOSE")
    print("One trade/session")
    print()

    # ========================================================================
    # LOAD
    # ========================================================================

    print("=" * 100)
    print("1. LOAD CANONICAL DATABENTO DATA")
    print("=" * 100)

    raw = load_databento_mnq()

    df = normalize_columns(raw)

    print(f"1-minute rows: {len(df):,}")
    print(f"Start: {df['timestamp ET'].min()}")
    print(f"End:   {df['timestamp ET'].max()}")

    # ========================================================================
    # M5
    # ========================================================================

    print()
    print("=" * 100)
    print("2. BUILD RTH M5")
    print("=" * 100)

    m5 = build_m5(df)

    print(f"Complete M5 bars: {len(m5):,}")
    print(f"Sessions: {m5['session_date'].nunique():,}")

    # ========================================================================
    # INDICATORS
    # ========================================================================

    print()
    print("=" * 100)
    print("3. BUILD EMA12 / EMA120 / ATR14")
    print("=" * 100)

    m5 = add_indicators(m5)

    # ========================================================================
    # AUDIT
    # ========================================================================

    variants = [
        (
            "A_intrabar_current",
            "INTRABAR",
            "INTRABAR",
            "CURRENT",
            True,
        ),
        (
            "B_intrabar_previous",
            "INTRABAR",
            "INTRABAR",
            "PREVIOUS",
            True,
        ),
        (
            "C_close_current",
            "CLOSE",
            "CLOSE",
            "CURRENT",
            True,
        ),
        (
            "D_close_previous",
            "CLOSE",
            "CLOSE",
            "PREVIOUS",
            True,
        ),
        (
            "E_intrabar_current_nextbar",
            "INTRABAR",
            "INTRABAR",
            "CURRENT",
            False,
        ),
        (
            "F_intrabar_previous_nextbar",
            "INTRABAR",
            "INTRABAR",
            "PREVIOUS",
            False,
        ),
        (
            "G_close_current_nextbar",
            "CLOSE",
            "CLOSE",
            "CURRENT",
            False,
        ),
        (
            "H_close_previous_nextbar",
            "CLOSE",
            "CLOSE",
            "PREVIOUS",
            False,
        ),
        (
            "I_intrabar_activation_close_exit",
            "INTRABAR",
            "CLOSE",
            "CURRENT",
            True,
        ),
        (
            "J_close_activation_intrabar_exit",
            "CLOSE",
            "INTRABAR",
            "CURRENT",
            True,
        ),
    ]

    all_results = []
    all_trades = []

    print()
    print("=" * 100)
    print("4. EXECUTION MECHANICS AUDIT")
    print("=" * 100)

    for n, (
        label,
        activation_mode,
        ema_exit_mode,
        ema_source,
        activation_bar_trail,
    ) in enumerate(variants, 1):
        print()
        print(f"[{n}/{len(variants)}] {label}")

        print(
            f"Activation={activation_mode} | "
            f"EMA exit={ema_exit_mode} | "
            f"EMA source={ema_source} | "
            f"same-bar trail={activation_bar_trail}"
        )

        trades = run_strategy(
            m5,
            activation_mode=activation_mode,
            ema_exit_mode=ema_exit_mode,
            ema_source=ema_source,
            activation_bar_trail=activation_bar_trail,
        )

        if trades.empty:
            print("[WARNING] No trades")
            continue

        trades["variant"] = label

        result = summarize(trades, label)

        all_results.append(result)

        all_trades.append(trades)

        print(
            f"Trades={result['trades']} | "
            f"Total R={result['total_R']:.4f} | "
            f"Exp={result['expectancy_R']:.5f}R | "
            f"WR={result['win_rate']:.4%} | "
            f"PF={result['profit_factor']:.4f} | "
            f"DD={result['max_drawdown_R']:.4f}R | "
            f"Sharpe={result['daily_sharpe']:.4f}"
        )

    # ========================================================================
    # RESULTS
    # ========================================================================

    results_df = pd.DataFrame(all_results)

    print()
    print("=" * 100)
    print("5. FULL SAMPLE RESULTS")
    print("=" * 100)

    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ========================================================================
    # EXIT REASON AUDIT
    # ========================================================================

    print()
    print("=" * 100)
    print("6. EXIT REASON AUDIT")
    print("=" * 100)

    for trades in all_trades:
        variant = trades["variant"].iloc[0]

        counts = trades["exit_reason"].value_counts().to_dict()

        print()
        print(variant)

        for reason, count in counts.items():
            print(f"  {reason:<20} {count:>6,}")

    # ========================================================================
    # ACTIVATION AUDIT
    # ========================================================================

    print()
    print("=" * 100)
    print("7. TRAIL ACTIVATION AUDIT")
    print("=" * 100)

    for trades in all_trades:
        variant = trades["variant"].iloc[0]

        activation_rate = trades["activated"].mean()

        print(f"{variant:<35} activated={activation_rate:.4%}")

    # ========================================================================
    # SAVE
    # ========================================================================

    output_dir = (
        PROJECT_ROOT
        / "src"
        / "research"
        / "results"
        / "session_momentum"
        / "parameter_audit"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "execution_mechanics_audit.csv"

    trades_path = output_dir / "execution_mechanics_trades.csv"

    results_df.to_csv(results_path, index=False)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False)

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print()
    print("Saved:")
    print(results_path)
    print(trades_path)


if __name__ == "__main__":
    main()
