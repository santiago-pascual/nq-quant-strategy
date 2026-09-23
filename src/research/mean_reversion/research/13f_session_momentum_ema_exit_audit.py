"""
SESSION MOMENTUM — EMA120 EXIT SEMANTICS AUDIT

FIXED
-----
- M5
- NY RTH
- Opening candle: 09:30–09:35 ET
- Entry: 09:35 ET CLOSE
- Direction:
    LONG  if opening close > EMA12
    SHORT if opening close < EMA12
- ATR14
- Initial stop = 8 * ATR14
- Trail activates at +0.5R using intrabar High/Low
- One trade per session
- No fixed TP

AUDITED
-------
1. CLOSE_CONDITION
   Exit when current close is on the wrong side of EMA120.

2. CROSS
   Exit only when price actually crosses EMA120:
       LONG:
           previous close >= previous EMA
           current close < current EMA

       SHORT:
           previous close <= previous EMA
           current close > current EMA

3. EMA_STOP_CURRENT
   EMA120 acts as a dynamic stop using current-bar EMA.

4. EMA_STOP_PREVIOUS
   EMA120 acts as a dynamic stop using previous completed bar EMA.

5. M5_CLOSE_CROSS
   Cross evaluated using M5 closes.

6. 1M_EMA_CROSS
   EMA120 calculated on 1-minute data and used to determine the
   first intrabar crossing after the +0.5R activation.

The purpose is to determine whether the large discrepancy between
our reproduction and the reference results comes from the exact
definition of "trail with EMA120".
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

ATR_PERIOD = 5
ATR_MULTIPLIER = 8.0

EMA_SIGNAL = 12
EMA_TRAIL = 120

TRAIL_TRIGGER_R = 0.5

RTH_START = (9, 30)
RTH_END = (16, 0)

ENTRY_TIME = (9, 35)


# ============================================================================
# LOAD
# ============================================================================

try:
    from src.databento_loader import load_databento_mnq
except Exception:
    from src.data_loader import load_data as load_databento_mnq


# ============================================================================
# DATA NORMALIZATION
# ============================================================================


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

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

    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(out.columns)}")

    ts = pd.to_datetime(out["timestamp ET"])

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")

    out["timestamp ET"] = ts.dt.tz_convert("America/New_York")

    return out.sort_values("timestamp ET").reset_index(drop=True)


# ============================================================================
# M5
# ============================================================================


def build_m5(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    ts = x["timestamp ET"]

    after_start = (ts.dt.hour > RTH_START[0]) | (
        (ts.dt.hour == RTH_START[0]) & (ts.dt.minute >= RTH_START[1])
    )

    before_end = (ts.dt.hour < RTH_END[0]) | (
        (ts.dt.hour == RTH_END[0]) & (ts.dt.minute < RTH_END[1])
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

    # Require all five one-minute observations.
    m5 = m5[m5["n_1m"] == 5].copy()

    m5["session_date"] = m5["timestamp ET"].dt.date

    return m5.reset_index(drop=True)


# ============================================================================
# INDICATORS
# ============================================================================


def add_m5_indicators(m5: pd.DataFrame) -> pd.DataFrame:

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

    x["tr"] = tr

    x["atr14"] = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

    return x


# ============================================================================
# 1-MINUTE EMA120
# ============================================================================


def add_1m_ema120(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    x["ema120_1m"] = x["close"].ewm(span=EMA_TRAIL, adjust=False).mean()

    x["session_date"] = x["timestamp ET"].dt.date

    return x


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

    return float((equity - peak).min())


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


def summarize(trades: pd.DataFrame, variant: str) -> dict:

    if trades.empty:
        return {
            "variant": variant,
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
        "variant": variant,
        "trades": len(trades),
        "total_R": r.sum(),
        "expectancy_R": r.mean(),
        "win_rate": (r > 0).mean(),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": max_drawdown(r),
        "daily_sharpe": daily_sharpe(trades),
    }


# ============================================================================
# TRADE ENGINE
# ============================================================================


def run_m5_strategy(
    m5: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:

    trades = []

    for session_date, session in m5.groupby("session_date", sort=True):
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

        if opening["close"] > opening["ema12"]:
            side = "LONG"

        elif opening["close"] < opening["ema12"]:
            side = "SHORT"

        else:
            continue

        entry_time = opening["timestamp ET"] + pd.Timedelta(minutes=5)

        if not (session["timestamp ET"] == entry_time).any():
            continue

        entry_price = float(opening["close"])

        atr = float(opening["atr14"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        risk = ATR_MULTIPLIER * atr

        if side == "LONG":
            initial_stop = entry_price - risk
        else:
            initial_stop = entry_price + risk

        start_idx = session.index[session["timestamp ET"] == entry_time][0]

        trade_data = session.loc[start_idx:].copy()

        activated = False
        activation_time = None

        exit_price = None
        exit_time = None
        exit_reason = None

        previous_close = None
        previous_ema = None

        for _, row in trade_data.iterrows():
            ts = row["timestamp ET"]

            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            ema = float(row["ema120"])

            # --------------------------------------------------------
            # Initial stop
            # --------------------------------------------------------

            if side == "LONG":
                if low <= initial_stop:
                    exit_price = initial_stop
                    exit_time = ts
                    exit_reason = "INITIAL_STOP"

                    break

            else:
                if high >= initial_stop:
                    exit_price = initial_stop
                    exit_time = ts
                    exit_reason = "INITIAL_STOP"

                    break

            # --------------------------------------------------------
            # +0.5R activation
            # Always intrabar in this audit.
            # --------------------------------------------------------

            trigger = TRAIL_TRIGGER_R * risk

            if not activated:
                if side == "LONG":
                    if high >= (entry_price + trigger):
                        activated = True
                        activation_time = ts

                else:
                    if low <= (entry_price - trigger):
                        activated = True
                        activation_time = ts

            # --------------------------------------------------------
            # EMA exit
            # --------------------------------------------------------

            if activated:
                # ====================================================
                # 1. CLOSE CONDITION
                # ====================================================

                if variant == "CLOSE_CONDITION":
                    if side == "LONG":
                        if close < ema:
                            exit_price = close
                            exit_time = ts
                            exit_reason = "EMA120_CLOSE"

                            break

                    else:
                        if close > ema:
                            exit_price = close
                            exit_time = ts
                            exit_reason = "EMA120_CLOSE"

                            break

                # ====================================================
                # 2. CROSS
                # ====================================================

                elif variant == "CROSS":
                    if previous_close is not None and previous_ema is not None:
                        if side == "LONG":
                            crossed = previous_close >= previous_ema and close < ema

                        else:
                            crossed = previous_close <= previous_ema and close > ema

                        if crossed:
                            exit_price = close
                            exit_time = ts
                            exit_reason = "EMA120_CROSS"

                            break

                # ====================================================
                # 3. CURRENT EMA STOP
                # ====================================================

                elif variant == "EMA_STOP_CURRENT":
                    if side == "LONG":
                        if low <= ema:
                            exit_price = ema
                            exit_time = ts
                            exit_reason = "EMA120_STOP"

                            break

                    else:
                        if high >= ema:
                            exit_price = ema
                            exit_time = ts
                            exit_reason = "EMA120_STOP"

                            break

                # ====================================================
                # 4. PREVIOUS EMA STOP
                # ====================================================

                elif variant == "EMA_STOP_PREVIOUS":
                    if previous_ema is not None:
                        if side == "LONG":
                            if low <= previous_ema:
                                exit_price = previous_ema

                                exit_time = ts
                                exit_reason = "EMA120_STOP_PREVIOUS"

                                break

                        else:
                            if high >= previous_ema:
                                exit_price = previous_ema

                                exit_time = ts
                                exit_reason = "EMA120_STOP_PREVIOUS"

                                break

                # ====================================================
                # 5. M5 CLOSE CROSS
                # ====================================================

                elif variant == "M5_CLOSE_CROSS":
                    if previous_close is not None and previous_ema is not None:
                        if side == "LONG":
                            crossed = previous_close >= previous_ema and close < ema

                        else:
                            crossed = previous_close <= previous_ema and close > ema

                        if crossed:
                            exit_price = close
                            exit_time = ts
                            exit_reason = "M5_CROSS"

                            break

            previous_close = close
            previous_ema = ema

        # ------------------------------------------------------------
        # End of data
        # ------------------------------------------------------------

        if exit_price is None:
            last = trade_data.iloc[-1]

            exit_price = float(last["close"])

            exit_time = last["timestamp ET"]

            exit_reason = "END_OF_DATA"

        # ------------------------------------------------------------
        # PNL
        # ------------------------------------------------------------

        if side == "LONG":
            pnl = exit_price - entry_price

        else:
            pnl = entry_price - exit_price

        net_R = pnl / risk

        bars_held = len(
            session[
                (session["timestamp ET"] >= entry_time)
                & (session["timestamp ET"] <= exit_time)
            ]
        )

        trades.append(
            {
                "variant": variant,
                "strategy": "SESSION_MOMENTUM",
                "side": side,
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "initial_stop": initial_stop,
                "risk_points": risk,
                "activated": activated,
                "activation_timestamp": activation_time,
                "exit_reason": exit_reason,
                "pnl_points": pnl,
                "net_R": net_R,
                "bars_held": bars_held,
            }
        )

    return pd.DataFrame(trades)


# ============================================================================
# 1-MINUTE CROSS AUDIT
# ============================================================================


def run_1m_cross_strategy(
    df1m: pd.DataFrame,
    m5: pd.DataFrame,
) -> pd.DataFrame:
    """
    EMA120 is calculated directly on 1-minute data.

    Entry / ATR / EMA12 remain M5 based.

    After +0.5R activation, the first 1-minute
    EMA120 cross closes the trade.
    """

    trades = []

    for session_date, session in m5.groupby("session_date", sort=True):
        session = session.sort_values("timestamp ET").reset_index(drop=True)

        opening = session[
            (session["timestamp ET"].dt.hour == 9)
            & (session["timestamp ET"].dt.minute == 30)
        ]

        if opening.empty:
            continue

        opening = opening.iloc[0]

        if opening["close"] > opening["ema12"]:
            side = "LONG"

        elif opening["close"] < opening["ema12"]:
            side = "SHORT"

        else:
            continue

        entry_time = opening["timestamp ET"] + pd.Timedelta(minutes=5)

        atr = float(opening["atr14"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        entry_price = float(opening["close"])

        risk = ATR_MULTIPLIER * atr

        if side == "LONG":
            initial_stop = entry_price - risk
        else:
            initial_stop = entry_price + risk

        day_1m = df1m[df1m["session_date"] == session_date].copy()

        day_1m = day_1m[day_1m["timestamp ET"] >= entry_time].copy()

        activated = False
        activation_time = None

        previous_close = None
        previous_ema = None

        exit_price = None
        exit_time = None
        exit_reason = None

        for _, row in day_1m.iterrows():
            ts = row["timestamp ET"]

            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            ema = float(row["ema120_1m"])

            # Initial stop
            if side == "LONG":
                if low <= initial_stop:
                    exit_price = initial_stop
                    exit_time = ts
                    exit_reason = "INITIAL_STOP"

                    break

            else:
                if high >= initial_stop:
                    exit_price = initial_stop
                    exit_time = ts
                    exit_reason = "INITIAL_STOP"

                    break

            # +0.5R activation
            trigger = TRAIL_TRIGGER_R * risk

            if not activated:
                if side == "LONG":
                    if high >= (entry_price + trigger):
                        activated = True
                        activation_time = ts

                else:
                    if low <= (entry_price - trigger):
                        activated = True
                        activation_time = ts

            # EMA120 1-minute cross
            if activated:
                if previous_close is not None and previous_ema is not None:
                    if side == "LONG":
                        crossed = previous_close >= previous_ema and close < ema

                    else:
                        crossed = previous_close <= previous_ema and close > ema

                    if crossed:
                        exit_price = close
                        exit_time = ts
                        exit_reason = "EMA120_1M_CROSS"

                        break

            previous_close = close
            previous_ema = ema

        if exit_price is None:
            if day_1m.empty:
                continue

            last = day_1m.iloc[-1]

            exit_price = float(last["close"])

            exit_time = last["timestamp ET"]

            exit_reason = "END_OF_DATA"

        if side == "LONG":
            pnl = exit_price - entry_price

        else:
            pnl = entry_price - exit_price

        net_R = pnl / risk

        trades.append(
            {
                "variant": "1M_EMA120_CROSS",
                "strategy": "SESSION_MOMENTUM",
                "side": side,
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "initial_stop": initial_stop,
                "risk_points": risk,
                "activated": activated,
                "activation_timestamp": activation_time,
                "exit_reason": exit_reason,
                "pnl_points": pnl,
                "net_R": net_R,
            }
        )

    return pd.DataFrame(trades)


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 100)
    print("SESSION MOMENTUM — EMA120 EXIT SEMANTICS AUDIT")
    print("=" * 100)

    print()
    print(f"ATR:              {ATR_PERIOD}")
    print(f"ATR multiplier:   {ATR_MULTIPLIER}")
    print(f"EMA signal:       {EMA_SIGNAL}")
    print(f"EMA trail:        {EMA_TRAIL}")
    print(f"Activation:       +{TRAIL_TRIGGER_R}R intrabar")
    print("Opening candle:   09:30–09:35 ET")
    print("Entry:            09:35 ET CLOSE")
    print("One trade/session")

    # ========================================================================
    # LOAD
    # ========================================================================

    print()
    print("=" * 100)
    print("1. LOAD CANONICAL DATA")
    print("=" * 100)

    raw = load_databento_mnq()

    df1m = normalize_columns(raw)

    print(f"1-minute rows: {len(df1m):,}")

    print(f"Start: {df1m['timestamp ET'].min()}")

    print(f"End:   {df1m['timestamp ET'].max()}")

    # ========================================================================
    # 1M EMA
    # ========================================================================

    df1m = add_1m_ema120(df1m)

    # ========================================================================
    # M5
    # ========================================================================

    print()
    print("=" * 100)
    print("2. BUILD M5")
    print("=" * 100)

    m5 = build_m5(df1m)

    m5 = add_m5_indicators(m5)

    print(f"M5 bars: {len(m5):,}")

    print(f"Sessions: {m5['session_date'].nunique():,}")

    # ========================================================================
    # M5 VARIANTS
    # ========================================================================

    variants = [
        "CLOSE_CONDITION",
        "CROSS",
        "EMA_STOP_CURRENT",
        "EMA_STOP_PREVIOUS",
        "M5_CLOSE_CROSS",
    ]

    results = []
    trade_frames = []

    print()
    print("=" * 100)
    print("3. M5 EMA120 EXIT SEMANTICS")
    print("=" * 100)

    for i, variant in enumerate(variants, 1):
        print()
        print(f"[{i}/{len(variants)}] {variant}")

        trades = run_m5_strategy(m5, variant)

        if trades.empty:
            print("[WARNING] No trades")
            continue

        summary = summarize(trades, variant)

        results.append(summary)

        trade_frames.append(trades)

        print(
            f"Trades={summary['trades']} | "
            f"Total R={summary['total_R']:.4f} | "
            f"Exp={summary['expectancy_R']:.5f}R | "
            f"WR={summary['win_rate']:.4%} | "
            f"PF={summary['profit_factor']:.4f} | "
            f"DD={summary['max_drawdown_R']:.4f}R | "
            f"Sharpe={summary['daily_sharpe']:.4f}"
        )

    # ========================================================================
    # 1M VARIANT
    # ========================================================================

    print()
    print("=" * 100)
    print("4. 1-MINUTE EMA120 CROSS")
    print("=" * 100)

    trades_1m = run_1m_cross_strategy(df1m, m5)

    if not trades_1m.empty:
        summary = summarize(trades_1m, "1M_EMA120_CROSS")

        results.append(summary)

        trade_frames.append(trades_1m)

        print(
            f"Trades={summary['trades']} | "
            f"Total R={summary['total_R']:.4f} | "
            f"Exp={summary['expectancy_R']:.5f}R | "
            f"WR={summary['win_rate']:.4%} | "
            f"PF={summary['profit_factor']:.4f} | "
            f"DD={summary['max_drawdown_R']:.4f}R | "
            f"Sharpe={summary['daily_sharpe']:.4f}"
        )

    # ========================================================================
    # RESULTS
    # ========================================================================

    results_df = pd.DataFrame(results)

    print()
    print("=" * 100)
    print("5. FINAL COMPARISON")
    print("=" * 100)

    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ========================================================================
    # EXIT REASONS
    # ========================================================================

    print()
    print("=" * 100)
    print("6. EXIT REASONS")
    print("=" * 100)

    for trades in trade_frames:
        variant = trades["variant"].iloc[0]

        print()
        print(variant)

        counts = trades["exit_reason"].value_counts()

        for reason, count in counts.items():
            print(f"  {reason:<25} {count:>6,}")

    # ========================================================================
    # SIDE
    # ========================================================================

    print()
    print("=" * 100)
    print("7. LONG / SHORT BREAKDOWN")
    print("=" * 100)

    side_results = []

    for trades in trade_frames:
        variant = trades["variant"].iloc[0]

        for side in ["LONG", "SHORT"]:
            subset = trades[trades["side"] == side]

            if subset.empty:
                continue

            r = subset["net_R"]

            side_results.append(
                {
                    "variant": variant,
                    "side": side,
                    "trades": len(subset),
                    "total_R": r.sum(),
                    "expectancy_R": r.mean(),
                    "win_rate": (r > 0).mean(),
                    "profit_factor": (profit_factor(r)),
                }
            )

    side_df = pd.DataFrame(side_results)

    print(side_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

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

    results_path = output_dir / "ema120_exit_semantics_audit.csv"

    trades_path = output_dir / "ema120_exit_semantics_trades.csv"

    side_path = output_dir / "ema120_exit_semantics_by_side.csv"

    results_df.to_csv(results_path, index=False)

    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)

    side_df.to_csv(side_path, index=False)

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print()
    print("Saved:")
    print(results_path)
    print(trades_path)
    print(side_path)


if __name__ == "__main__":
    main()
