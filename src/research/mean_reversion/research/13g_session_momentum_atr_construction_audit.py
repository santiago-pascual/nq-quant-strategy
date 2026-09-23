"""
SESSION MOMENTUM — ATR CONSTRUCTION AUDIT

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
- EMA12: standard EMA
- EMA120: standard EMA
- Initial stop: ATR * 8
- Trail activation: +0.5R using intrabar High/Low
- EMA120 trailing: M5 close condition
- One trade per session
- No fixed TP

ONLY ATR CONSTRUCTION VARIES
----------------------------
ATR periods:
    5, 7, 10, 12, 14, 16, 20, 24, 30, 40

ATR methods:
    RMA / Wilder
    EMA
    SMA

IMPORTANT
---------
The source only specifies "8 × ATR".
It does NOT explicitly specify:
- ATR period
- ATR smoothing
- ATR initialization

Therefore this is an audit, not a parameter optimization.

All variants use the exact same:
- market data
- M5 construction
- EMA12
- EMA120
- entry
- +0.5R activation
- EMA120 exit semantics
- one-trade-per-session rule
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIG
# ============================================================================

ATR_PERIODS = [
    5,
    7,
    10,
    12,
    14,
    16,
    20,
    24,
    30,
    40,
]

ATR_METHODS = [
    "RMA",
    "EMA",
    "SMA",
]

ATR_MULTIPLIER = 8.0

EMA_SIGNAL = 12
EMA_TRAIL = 120

TRAIL_TRIGGER_R = 0.5

RTH_START_HOUR = 9
RTH_START_MINUTE = 30

RTH_END_HOUR = 16
RTH_END_MINUTE = 0


# ============================================================================
# LOADER
# ============================================================================

try:
    from src.databento_loader import load_databento_mnq
except Exception:
    from src.data_loader import load_data as load_databento_mnq


# ============================================================================
# NORMALIZE DATA
# ============================================================================


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    rename = {}

    for c in out.columns:
        lc = str(c).lower().strip()

        if lc in {
            "timestamp et",
            "timestamp_et",
            "timestamp",
        }:
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

    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nAvailable columns: {list(out.columns)}"
        )

    ts = pd.to_datetime(out["timestamp ET"])

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")

    out["timestamp ET"] = ts.dt.tz_convert("America/New_York")

    return out.sort_values("timestamp ET").reset_index(drop=True)


# ============================================================================
# BUILD M5
# ============================================================================


def build_m5(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    ts = x["timestamp ET"]

    after_start = (ts.dt.hour > RTH_START_HOUR) | (
        (ts.dt.hour == RTH_START_HOUR) & (ts.dt.minute >= RTH_START_MINUTE)
    )

    before_end = (ts.dt.hour < RTH_END_HOUR) | (
        (ts.dt.hour == RTH_END_HOUR) & (ts.dt.minute < RTH_END_MINUTE)
    )

    x = x[after_start & before_end].copy()

    x["bar_time"] = x["timestamp ET"].dt.floor("5min")

    agg = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "n_1m": ("close", "size"),
    }

    if "volume" in x.columns:
        agg["volume"] = ("volume", "sum")

    m5 = (
        x.groupby("bar_time", sort=True)
        .agg(**agg)
        .reset_index()
        .rename(columns={"bar_time": "timestamp ET"})
    )

    # Complete five-minute candle.
    m5 = m5[m5["n_1m"] == 5].copy()

    m5["session_date"] = m5["timestamp ET"].dt.date

    return m5.reset_index(drop=True)


# ============================================================================
# BASE INDICATORS
# ============================================================================


def add_base_indicators(m5: pd.DataFrame) -> pd.DataFrame:

    x = m5.copy()

    # EMA12 signal
    x["ema12"] = x["close"].ewm(span=EMA_SIGNAL, adjust=False).mean()

    # EMA120 trailing
    x["ema120"] = x["close"].ewm(span=EMA_TRAIL, adjust=False).mean()

    # True Range
    prev_close = x["close"].shift(1)

    tr1 = x["high"] - x["low"]

    tr2 = (x["high"] - prev_close).abs()

    tr3 = (x["low"] - prev_close).abs()

    x["tr"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return x


# ============================================================================
# ATR
# ============================================================================


def calculate_atr(
    tr: pd.Series,
    period: int,
    method: str,
) -> pd.Series:

    method = method.upper()

    if method == "RMA":
        # Wilder / RMA
        return tr.ewm(
            alpha=1.0 / period,
            adjust=False,
        ).mean()

    elif method == "EMA":
        return tr.ewm(
            span=period,
            adjust=False,
        ).mean()

    elif method == "SMA":
        return tr.rolling(
            window=period,
            min_periods=period,
        ).mean()

    else:
        raise ValueError(f"Unknown ATR method: {method}")


# ============================================================================
# METRICS
# ============================================================================


def profit_factor(r: pd.Series) -> float:

    gross_profit = r[r > 0].sum()

    gross_loss = -r[r < 0].sum()

    if gross_loss == 0:
        return np.nan

    return float(gross_profit / gross_loss)


def max_drawdown(r: pd.Series) -> float:

    equity = r.cumsum()

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


def summarize(
    trades: pd.DataFrame,
    atr_period: int,
    atr_method: str,
) -> dict:

    if trades.empty:
        return {
            "atr_period": atr_period,
            "atr_method": atr_method,
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
        "atr_period": atr_period,
        "atr_method": atr_method,
        "trades": len(trades),
        "total_R": r.sum(),
        "expectancy_R": r.mean(),
        "win_rate": (r > 0).mean(),
        "profit_factor": (profit_factor(r)),
        "max_drawdown_R": (max_drawdown(r)),
        "daily_sharpe": (daily_sharpe(trades)),
    }


# ============================================================================
# STRATEGY ENGINE
# ============================================================================


def run_strategy(
    df: pd.DataFrame,
    atr_column: str,
    atr_period: int,
    atr_method: str,
) -> pd.DataFrame:

    trades = []

    for session_date, session in df.groupby("session_date", sort=True):
        session = session.sort_values("timestamp ET").reset_index(drop=True)

        # ------------------------------------------------------------
        # Opening candle
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Entry
        # ------------------------------------------------------------

        entry_time = opening["timestamp ET"] + pd.Timedelta(minutes=5)

        if not (session["timestamp ET"] == entry_time).any():
            continue

        entry_price = float(opening["close"])

        atr = float(opening[atr_column])

        if not np.isfinite(atr) or atr <= 0:
            continue

        risk = ATR_MULTIPLIER * atr

        if side == "LONG":
            initial_stop = entry_price - risk

        else:
            initial_stop = entry_price + risk

        # Entry index
        start_idx = session.index[session["timestamp ET"] == entry_time][0]

        trade_data = session.loc[start_idx:].copy()

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------

        activated = False
        activation_timestamp = None

        exit_price = None
        exit_timestamp = None
        exit_reason = None

        # ------------------------------------------------------------
        # Trade loop
        # ------------------------------------------------------------

        for _, row in trade_data.iterrows():
            ts = row["timestamp ET"]

            high = float(row["high"])

            low = float(row["low"])

            close = float(row["close"])

            ema120 = float(row["ema120"])

            # ========================================================
            # INITIAL STOP
            # ========================================================

            if side == "LONG":
                if low <= initial_stop:
                    exit_price = initial_stop

                    exit_timestamp = ts

                    exit_reason = "INITIAL_STOP"

                    break

            else:
                if high >= initial_stop:
                    exit_price = initial_stop

                    exit_timestamp = ts

                    exit_reason = "INITIAL_STOP"

                    break

            # ========================================================
            # +0.5R ACTIVATION
            # ========================================================

            if not activated:
                trigger = TRAIL_TRIGGER_R * risk

                if side == "LONG":
                    if high >= (entry_price + trigger):
                        activated = True

                        activation_timestamp = ts

                else:
                    if low <= (entry_price - trigger):
                        activated = True

                        activation_timestamp = ts

            # ========================================================
            # EMA120 TRAILING
            #
            # Fixed semantics for this audit:
            # after activation, exit when M5 close crosses
            # to the wrong side of EMA120.
            #
            # This keeps exit semantics constant while ATR varies.
            # ========================================================

            if activated:
                if side == "LONG":
                    if close < ema120:
                        exit_price = close

                        exit_timestamp = ts

                        exit_reason = "EMA120_CLOSE"

                        break

                else:
                    if close > ema120:
                        exit_price = close

                        exit_timestamp = ts

                        exit_reason = "EMA120_CLOSE"

                        break

        # ============================================================
        # END OF DATA
        # ============================================================

        if exit_price is None:
            last = trade_data.iloc[-1]

            exit_price = float(last["close"])

            exit_timestamp = last["timestamp ET"]

            exit_reason = "END_OF_DATA"

        # ============================================================
        # PNL
        # ============================================================

        if side == "LONG":
            pnl_points = exit_price - entry_price

        else:
            pnl_points = entry_price - exit_price

        net_R = pnl_points / risk

        bars_held = len(
            session[
                (session["timestamp ET"] >= entry_time)
                & (session["timestamp ET"] <= exit_timestamp)
            ]
        )

        trades.append(
            {
                "atr_period": atr_period,
                "atr_method": atr_method,
                "strategy": "SESSION_MOMENTUM",
                "side": side,
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "atr_at_entry": atr,
                "risk_points": risk,
                "initial_stop": initial_stop,
                "activated": activated,
                "activation_timestamp": activation_timestamp,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "net_R": net_R,
                "bars_held": bars_held,
            }
        )

    return pd.DataFrame(trades)


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 100)
    print("SESSION MOMENTUM — ATR CONSTRUCTION AUDIT")
    print("=" * 100)

    print()
    print(f"ATR multiplier: {ATR_MULTIPLIER}")

    print(f"ATR periods:    {ATR_PERIODS}")

    print(f"ATR methods:    {ATR_METHODS}")

    print("Entry:          09:35 ET CLOSE")

    print("Activation:     +0.5R INTRABAR")

    print("EMA120 exit:    M5 CLOSE CONDITION")

    # ========================================================================
    # LOAD
    # ========================================================================

    print()
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
    print("2. BUILD M5")
    print("=" * 100)

    m5 = build_m5(df)

    print(f"Complete M5 bars: {len(m5):,}")

    print(f"Sessions: {m5['session_date'].nunique():,}")

    # ========================================================================
    # BASE INDICATORS
    # ========================================================================

    print()
    print("=" * 100)
    print("3. BUILD EMA12 / EMA120 / TRUE RANGE")
    print("=" * 100)

    m5 = add_base_indicators(m5)

    # ========================================================================
    # CREATE ALL ATR SERIES
    # ========================================================================

    print()
    print("=" * 100)
    print("4. BUILD ATR GRID")
    print("=" * 100)

    atr_columns = {}

    for method in ATR_METHODS:
        for period in ATR_PERIODS:
            column = f"atr_{method.lower()}_{period}"

            m5[column] = calculate_atr(
                m5["tr"],
                period,
                method,
            )

            atr_columns[(method, period)] = column

            print(f"Built {column}")

    # ========================================================================
    # FULL SAMPLE
    # ========================================================================

    print()
    print("=" * 100)
    print("5. RUN ATR GRID")
    print("=" * 100)

    results = []

    trade_frames = []

    total = len(ATR_METHODS) * len(ATR_PERIODS)

    counter = 0

    for method in ATR_METHODS:
        for period in ATR_PERIODS:
            counter += 1

            column = atr_columns[(method, period)]

            print()
            print(f"[{counter}/{total}] {method} ATR{period}")

            trades = run_strategy(
                m5,
                column,
                period,
                method,
            )

            if trades.empty:
                print("[WARNING] No trades")

                continue

            summary = summarize(
                trades,
                period,
                method,
            )

            results.append(summary)

            trade_frames.append(trades)

            print(
                f"Trades="
                f"{summary['trades']} | "
                f"Total R="
                f"{summary['total_R']:.4f} | "
                f"Exp="
                f"{summary['expectancy_R']:.5f}R | "
                f"WR="
                f"{summary['win_rate']:.4%} | "
                f"PF="
                f"{summary['profit_factor']:.4f} | "
                f"DD="
                f"{summary['max_drawdown_R']:.4f}R | "
                f"Sharpe="
                f"{summary['daily_sharpe']:.4f}"
            )

    results_df = pd.DataFrame(results)

    # ========================================================================
    # SORT FOR DISPLAY
    # ========================================================================

    print()
    print("=" * 100)
    print("6. FULL ATR COMPARISON")
    print("=" * 100)

    display_df = results_df.sort_values(
        [
            "atr_method",
            "atr_period",
        ]
    ).reset_index(drop=True)

    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ========================================================================
    # METHOD SUMMARY
    # ========================================================================

    print()
    print("=" * 100)
    print("7. METHOD SUMMARY")
    print("=" * 100)

    for method in ATR_METHODS:
        subset = results_df[results_df["atr_method"] == method]

        print()
        print(f"{method}")

        print(
            subset[
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
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )

    # ========================================================================
    # LONG / SHORT FOR EVERY VARIANT
    # ========================================================================

    print()
    print("=" * 100)
    print("8. LONG / SHORT BREAKDOWN")
    print("=" * 100)

    side_results = []

    for trades in trade_frames:
        if trades.empty:
            continue

        method = trades["atr_method"].iloc[0]

        period = int(trades["atr_period"].iloc[0])

        for side in [
            "LONG",
            "SHORT",
        ]:
            subset = trades[trades["side"] == side]

            if subset.empty:
                continue

            r = subset["net_R"]

            side_results.append(
                {
                    "atr_method": method,
                    "atr_period": period,
                    "side": side,
                    "trades": len(subset),
                    "total_R": r.sum(),
                    "expectancy_R": r.mean(),
                    "win_rate": (r > 0).mean(),
                    "profit_factor": profit_factor(r),
                }
            )

    side_df = pd.DataFrame(side_results)

    print(side_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ========================================================================
    # EXIT REASONS
    # ========================================================================

    print()
    print("=" * 100)
    print("9. EXIT REASON COUNTS")
    print("=" * 100)

    exit_results = []

    for trades in trade_frames:
        if trades.empty:
            continue

        method = trades["atr_method"].iloc[0]

        period = int(trades["atr_period"].iloc[0])

        counts = trades["exit_reason"].value_counts()

        for reason, count in counts.items():
            exit_results.append(
                {
                    "atr_method": method,
                    "atr_period": period,
                    "exit_reason": reason,
                    "count": int(count),
                }
            )

    exit_df = pd.DataFrame(exit_results)

    print(exit_df.to_string(index=False))

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

    results_path = output_dir / "atr_construction_audit.csv"

    trades_path = output_dir / "atr_construction_trades.csv"

    side_path = output_dir / "atr_construction_by_side.csv"

    exits_path = output_dir / "atr_construction_exit_reasons.csv"

    results_df.to_csv(results_path, index=False)

    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)

    side_df.to_csv(side_path, index=False)

    exit_df.to_csv(exits_path, index=False)

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print()
    print("Saved:")
    print(results_path)
    print(trades_path)
    print(side_path)
    print(exits_path)


if __name__ == "__main__":
    main()
