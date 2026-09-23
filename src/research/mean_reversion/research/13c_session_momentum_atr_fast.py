"""
13C — SESSION MOMENTUM: FAST ATR / OPENING-CANDLE AUDIT

This script isolates the two main reconstruction uncertainties:

1. ATR lookback period.
2. Opening-candle / entry convention.

Source-defined rules:
    M5
    NY session open
    EMA(12)
    EMA(120) trailing stop
    8 ATR initial stop
    trail activates at +0.5R
    one trade per session
    no fixed target

Current reconstruction:
    Signal candle = 09:30 -> 09:35 ET
    Direction = opening-candle CLOSE vs EMA12
    Entry = 09:35 ET, at the CLOSE of the opening candle
    Initial stop = 8 * ATR measured on opening candle
    ATR = Wilder/RMA-style EMA of True Range
    EMA120 trailing stop updated bar-by-bar
    No forced session close
    Position may carry forward

IMPORTANT
---------
This script intentionally DOES NOT calculate HMM/VOL context.

HMM/VOL analysis will be performed after the ATR/opening mechanics
are understood.

No robustness testing.
No Monte Carlo.
No walk-forward optimization.
No automatic ATR selection.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from src.databento_loader import load_databento_mnq
from src.session_engine import add_session_information


# =============================================================================
# CONFIG
# =============================================================================

EMA_SIGNAL = 12
EMA_TRAIL = 120

ATR_PERIOD = 14

ATR_AUDIT_PERIODS = [
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

ATR_MULTIPLIER = 8.0

TRAIL_ACTIVATION_R = 0.5

NY_OPEN_HOUR = 9
NY_OPEN_MINUTE = 30

RTH_END_HOUR = 16
RTH_END_MINUTE = 0

ANALYSIS_START = pd.Timestamp(
    "2020-01-01",
    tz="America/New_York",
)

IS_END = pd.Timestamp(
    "2024-12-31 23:59:59",
    tz="America/New_York",
)


# =============================================================================
# REFERENCE VALUES
# =============================================================================

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
# OUTPUT
# =============================================================================

RESULTS_ROOT = ROOT / "src" / "research" / "results" / "session_momentum"

AUDIT_DIR = RESULTS_ROOT / "parameter_audit"

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# UTILITIES
# =============================================================================


def banner(title: str) -> None:

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def max_drawdown(
    equity: pd.Series,
) -> float:

    if equity.empty:
        return np.nan

    peak = equity.cummax()

    return float((equity - peak).min())


def profit_factor(
    r: pd.Series,
) -> float:

    r = r.dropna().astype(float)

    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()

    if losses == 0:
        if wins > 0:
            return np.inf

        return np.nan

    return float(wins / losses)


def streaks(
    r: pd.Series,
) -> tuple[int, int]:

    best_w = 0
    best_l = 0

    current_w = 0
    current_l = 0

    for value in r:
        if value > 0:
            current_w += 1
            current_l = 0

            best_w = max(
                best_w,
                current_w,
            )

        elif value < 0:
            current_l += 1
            current_w = 0

            best_l = max(
                best_l,
                current_l,
            )

        else:
            current_w = 0
            current_l = 0

    return best_w, best_l


def t_stat(
    r: pd.Series,
) -> float:

    r = r.dropna().astype(float)

    if len(r) < 2:
        return np.nan

    std = r.std(ddof=1)

    if std == 0:
        return np.nan

    return float(r.mean() / (std / np.sqrt(len(r))))


def daily_sharpe(
    trades: pd.DataFrame,
) -> float:

    if trades.empty:
        return np.nan

    daily = trades.groupby(trades["entry_timestamp"].dt.date)["r"].sum()

    if len(daily) < 2:
        return np.nan

    std = daily.std(ddof=1)

    if std == 0:
        return np.nan

    return float(np.sqrt(252.0) * daily.mean() / std)


def daily_sortino(
    trades: pd.DataFrame,
) -> float:

    if trades.empty:
        return np.nan

    daily = trades.groupby(trades["entry_timestamp"].dt.date)["r"].sum()

    if len(daily) < 2:
        return np.nan

    downside = daily[daily < 0]

    if downside.empty:
        return np.inf

    downside_dev = np.sqrt(np.mean(np.square(downside)))

    if downside_dev == 0:
        return np.inf

    return float(np.sqrt(252.0) * daily.mean() / downside_dev)


# =============================================================================
# MARKET DATA
# =============================================================================


def load_market() -> pd.DataFrame:

    banner("1. LOAD CANONICAL DATABENTO DATA")

    df = load_databento_mnq().copy()

    ensure_columns(
        df,
        [
            "timestamp ET",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
        "Databento market",
    )

    # The canonical loader already supplies timestamp ET.
    ts = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")

    else:
        ts = ts.dt.tz_convert("America/New_York")

    df["timestamp_et"] = ts

    df = df.sort_values("timestamp_et").reset_index(drop=True)

    print(f"1-minute rows: {len(df):,}")

    print(f"Start: {df['timestamp_et'].iloc[0]}")

    print(f"End:   {df['timestamp_et'].iloc[-1]}")

    # Canonical project session classification.
    df = add_session_information(df)

    if "market_period" in df.columns:
        df = df.loc[df["market_period"].eq("RTH")].copy()

    # Explicit NY RTH window.
    t = df["timestamp_et"].dt.time

    start = pd.Timestamp("09:30").time()

    end = pd.Timestamp("16:00").time()

    df = df.loc[(t >= start) & (t < end)].copy()

    return df


# =============================================================================
# M5
# =============================================================================


def resample_m5(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    banner("2. BUILD RTH 5-MINUTE BARS")

    x = rth.copy().set_index("timestamp_et")

    # RTH begins exactly at 09:30 ET.
    x["bar"] = x.index.floor("5min")

    m5 = (
        x.groupby(
            "bar",
            sort=True,
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            minute_count=(
                "close",
                "count",
            ),
        )
        .reset_index()
        .rename(columns={"bar": "timestamp_et"})
    )

    # Only complete M5 bars.
    m5 = m5.loc[m5["minute_count"].eq(5)].copy()

    m5 = m5.drop(columns=["minute_count"])

    m5["session_date"] = m5["timestamp_et"].dt.date

    m5 = m5.sort_values("timestamp_et").reset_index(drop=True)

    print(f"Complete RTH M5 bars: {len(m5):,}")

    print(f"Sessions: {m5['session_date'].nunique():,}")

    return m5


# =============================================================================
# OPENING CANDLE AUDIT
# =============================================================================


def opening_candle_audit(
    m5: pd.DataFrame,
) -> pd.DataFrame:

    banner("OPENING-CANDLE AUDIT")

    sessions = m5["session_date"].nunique()

    opening = m5.loc[
        (m5["timestamp_et"].dt.hour == NY_OPEN_HOUR)
        & (m5["timestamp_et"].dt.minute == NY_OPEN_MINUTE)
    ].copy()

    bad = opening.loc[~(opening["timestamp_et"].dt.minute.eq(NY_OPEN_MINUTE))]

    sessions_with_open = opening["session_date"].nunique()

    print("Expected signal candle: 09:30–09:35 America/New_York")

    print("Expected entry time: 09:35 America/New_York")

    print("Expected entry price: CLOSE of 09:30–09:35 candle")

    print(f"Sessions in M5 dataset: {sessions:,}")

    print(f"Sessions with 09:30 bar: {sessions_with_open:,}")

    print(f"Sessions without 09:30 bar: {sessions - sessions_with_open:,}")

    print(f"09:30 bars found: {len(opening):,}")

    print(f"Bad opening minute labels: {len(bad):,}")

    result = pd.DataFrame(
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
                sessions_with_open,
                sessions - sessions_with_open,
                len(opening),
                len(bad),
            ],
        }
    )

    return result


# =============================================================================
# INDICATORS
# =============================================================================


def add_indicators(
    m5: pd.DataFrame,
) -> pd.DataFrame:

    banner("3. BUILD EMA(12), EMA(120), TRUE RANGE + ATR GRID")

    x = m5.copy()

    x["ema12"] = (
        x["close"]
        .ewm(
            span=EMA_SIGNAL,
            adjust=False,
        )
        .mean()
    )

    x["ema120"] = (
        x["close"]
        .ewm(
            span=EMA_TRAIL,
            adjust=False,
        )
        .mean()
    )

    prev_close = x["close"].shift(1)

    x["true_range"] = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Precompute every ATR exactly once.
    # Wilder/RMA-style ATR:
    # alpha = 1 / period
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

    print("EMA12: calculated")

    print("EMA120: calculated")

    print("True Range: calculated")

    print(f"ATR grid: {ATR_AUDIT_PERIODS}")

    return x


# =============================================================================
# POSITION
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

    activated: bool = False

    trail_stop: float | None = None

    activation_time: pd.Timestamp | None = None


# =============================================================================
# TRADE FINALIZATION
# =============================================================================


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
    }


# =============================================================================
# STRATEGY
# =============================================================================


def run_strategy(
    m5: pd.DataFrame,
    atr_period: int,
) -> pd.DataFrame:

    atr_col = f"atr_{atr_period}"

    if atr_col not in m5.columns:
        raise KeyError(f"Missing ATR column: {atr_col}")

    x = m5.copy().sort_values("timestamp_et").reset_index(drop=True)

    x["atr"] = x[atr_col]

    trades: list[dict] = []

    pos: Position | None = None

    sessions_with_entry: set = set()

    # -------------------------------------------------------------------------
    # Iterate through M5 bars.
    # -------------------------------------------------------------------------

    for _, row in x.iterrows():
        ts = row["timestamp_et"]

        # =====================================================================
        # 1. MANAGE EXISTING POSITION
        # =====================================================================

        if pos is not None:
            if pos.side == "LONG":
                current_stop = (
                    pos.trail_stop
                    if (pos.activated and pos.trail_stop is not None)
                    else pos.initial_stop
                )

                # Stop check.
                if row["low"] <= current_stop:
                    trades.append(
                        finalize_trade(
                            pos,
                            ts,
                            float(current_stop),
                            ("TRAIL_STOP" if pos.activated else "INITIAL_STOP"),
                        )
                    )

                    pos = None

                    continue

                # +0.5R activation.
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
                current_stop = (
                    pos.trail_stop
                    if (pos.activated and pos.trail_stop is not None)
                    else pos.initial_stop
                )

                # Stop check.
                if row["high"] >= current_stop:
                    trades.append(
                        finalize_trade(
                            pos,
                            ts,
                            float(current_stop),
                            ("TRAIL_STOP" if pos.activated else "INITIAL_STOP"),
                        )
                    )

                    pos = None

                    continue

                # +0.5R activation for SHORT.
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

            # Position remains open.
            continue

        # =====================================================================
        # 2. FIND FIRST NY OPENING CANDLE
        # =====================================================================

        if ts.hour != NY_OPEN_HOUR or ts.minute != NY_OPEN_MINUTE:
            continue

        session = row["session_date"]

        if session in sessions_with_entry:
            continue

        # Need valid indicators.
        if (
            not np.isfinite(row["ema12"])
            or not np.isfinite(row["ema120"])
            or not np.isfinite(row["atr"])
            or row["atr"] <= 0
        ):
            continue

        # =====================================================================
        # 3. SIGNAL
        # =====================================================================

        if row["close"] > row["ema12"]:
            side = "LONG"

        elif row["close"] < row["ema12"]:
            side = "SHORT"

        else:
            continue

        # =====================================================================
        # 4. ENTRY
        #
        # IMPORTANT:
        #
        # The opening candle is 09:30 -> 09:35.
        #
        # Entry price = CLOSE of that same candle.
        #
        # We timestamp the entry at 09:35.
        # =====================================================================

        entry_price = float(row["close"])

        entry_time = ts + pd.Timedelta(minutes=5)

        # =====================================================================
        # 5. INITIAL STOP
        # =====================================================================

        risk_points = ATR_MULTIPLIER * float(row["atr"])

        if risk_points <= 0 or not np.isfinite(risk_points):
            continue

        if side == "LONG":
            initial_stop = entry_price - risk_points

        else:
            initial_stop = entry_price + risk_points

        pos = Position(
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            risk_points=risk_points,
            initial_stop=initial_stop,
            signal_time=ts,
            signal_ema12=float(row["ema12"]),
            signal_ema120=float(row["ema120"]),
            signal_atr=float(row["atr"]),
        )

        sessions_with_entry.add(session)

    # =====================================================================
    # END OF DATA
    # =====================================================================

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
        return out

    out["signal_timestamp"] = pd.to_datetime(
        out["signal_timestamp"],
        utc=True,
    )

    out["entry_timestamp"] = pd.to_datetime(
        out["entry_timestamp"],
        utc=True,
    )

    out["exit_timestamp"] = pd.to_datetime(
        out["exit_timestamp"],
        utc=True,
    )

    return out.sort_values("entry_timestamp").reset_index(drop=True)


# =============================================================================
# METRICS
# =============================================================================


def metrics(
    trades: pd.DataFrame,
) -> dict:

    if trades.empty:
        return {
            "trades": 0,
            "total_R": np.nan,
            "expectancy_R": np.nan,
            "median_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "average_win_R": np.nan,
            "average_loss_R": np.nan,
            "payoff_ratio": np.nan,
            "t_stat": np.nan,
            "daily_sharpe": np.nan,
            "daily_sortino": np.nan,
        }

    r = trades["r"].astype(float)

    equity = r.cumsum()

    wins = r[r > 0]
    losses = r[r < 0]

    best_w, best_l = streaks(r)

    payoff = np.nan

    if not wins.empty and not losses.empty:
        payoff = wins.mean() / abs(losses.mean())

    return {
        "trades": len(r),
        "total_R": float(r.sum()),
        "expectancy_R": float(r.mean()),
        "median_R": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": max_drawdown(equity),
        "average_win_R": (float(wins.mean()) if not wins.empty else np.nan),
        "average_loss_R": (float(losses.mean()) if not losses.empty else np.nan),
        "payoff_ratio": payoff,
        "t_stat": t_stat(r),
        "daily_sharpe": daily_sharpe(trades),
        "daily_sortino": daily_sortino(trades),
        "longest_win_streak": best_w,
        "longest_loss_streak": best_l,
        "trail_activation_pct": float(trades["trail_activated"].mean()),
    }


# =============================================================================
# ATR AUDIT
# =============================================================================


def run_atr_audit(
    m5: pd.DataFrame,
) -> pd.DataFrame:

    banner("4. FAST ATR PERIOD AUDIT")

    print(f"Periods tested: {ATR_AUDIT_PERIODS}")

    print("Opening candle: 09:30–09:35 ET")

    print("Entry: 09:35 ET CLOSE")

    print("EMA12 / EMA120 / TR / ATRs already calculated once.")

    rows = []

    total = len(ATR_AUDIT_PERIODS)

    for n, period in enumerate(
        ATR_AUDIT_PERIODS,
        start=1,
    ):
        start_time = time.perf_counter()

        print(
            f"\n[{n}/{total}] Running ATR({period}) ...",
            flush=True,
        )

        trades = run_strategy(
            m5,
            atr_period=period,
        )

        if trades.empty:
            print(f"[DONE] ATR({period}) -> ZERO TRADES")

            continue

        # ---------------------------------------------------------------------
        # Analysis period.
        # ---------------------------------------------------------------------

        trades = trades.loc[
            trades["signal_timestamp"] >= ANALYSIS_START.tz_convert("UTC")
        ].copy()

        trades["sample"] = np.where(
            trades["signal_timestamp"] <= IS_END.tz_convert("UTC"),
            "IS",
            "OOS",
        )

        full = metrics(trades)

        is_metrics = metrics(trades.loc[trades["sample"].eq("IS")])

        oos_metrics = metrics(trades.loc[trades["sample"].eq("OOS")])

        elapsed = time.perf_counter() - start_time

        row = {
            "atr_period": period,
            # Full.
            "trades": full["trades"],
            "total_R": full["total_R"],
            "expectancy_R": full["expectancy_R"],
            "median_R": full["median_R"],
            "win_rate": full["win_rate"],
            "profit_factor": full["profit_factor"],
            "max_drawdown_R": full["max_drawdown_R"],
            "daily_sharpe": full["daily_sharpe"],
            "daily_sortino": full["daily_sortino"],
            "t_stat": full["t_stat"],
            # IS.
            "is_trades": is_metrics["trades"],
            "is_total_R": is_metrics["total_R"],
            "is_expectancy_R": is_metrics["expectancy_R"],
            "is_win_rate": is_metrics["win_rate"],
            "is_profit_factor": is_metrics["profit_factor"],
            "is_max_drawdown_R": is_metrics["max_drawdown_R"],
            # OOS.
            "oos_trades": oos_metrics["trades"],
            "oos_total_R": oos_metrics["total_R"],
            "oos_expectancy_R": oos_metrics["expectancy_R"],
            "oos_win_rate": oos_metrics["win_rate"],
            "oos_profit_factor": oos_metrics["profit_factor"],
            "oos_max_drawdown_R": oos_metrics["max_drawdown_R"],
            "oos_daily_sharpe": oos_metrics["daily_sharpe"],
            "oos_t_stat": oos_metrics["t_stat"],
            "seconds": elapsed,
        }

        rows.append(row)

        # Save individual trade stream.
        trades.to_csv(
            AUDIT_DIR / f"trades_atr_{period}.csv",
            index=False,
        )

        print(
            f"[DONE] ATR({period}) | "
            f"Trades={len(trades):,} | "
            f"PF={full['profit_factor']:.4f} | "
            f"WR={full['win_rate']:.2%} | "
            f"Exp={full['expectancy_R']:.5f}R | "
            f"OOS PF={oos_metrics['profit_factor']:.4f} | "
            f"{elapsed:.2f}s",
            flush=True,
        )

    result = pd.DataFrame(rows)

    # Save master result.
    result.to_csv(
        AUDIT_DIR / "atr_period_audit.csv",
        index=False,
    )

    return result


# =============================================================================
# REPORT
# =============================================================================


def print_audit(
    result: pd.DataFrame,
) -> None:

    banner("ATR AUDIT — FULL SAMPLE")

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

    banner("ATR AUDIT — IS")

    print(
        result[
            [
                "atr_period",
                "is_trades",
                "is_total_R",
                "is_expectancy_R",
                "is_win_rate",
                "is_profit_factor",
                "is_max_drawdown_R",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    banner("ATR AUDIT — OOS")

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


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("SESSION MOMENTUM — FAST ATR AUDIT")

    print(f"Project root: {ROOT}")

    print(f"Analysis start: {ANALYSIS_START}")

    print(f"IS end:         {IS_END}")

    print(f"ATR periods:    {ATR_AUDIT_PERIODS}")

    print(f"ATR multiplier: {ATR_MULTIPLIER}")

    print(f"EMA signal:     {EMA_SIGNAL}")

    print(f"EMA trail:      {EMA_TRAIL}")

    print(f"Trail trigger:  +{TRAIL_ACTIVATION_R}R")

    print("Opening candle: 09:30–09:35 ET")

    print("Entry:          09:35 ET CLOSE")

    # -------------------------------------------------------------------------
    # Expensive work — exactly once.
    # -------------------------------------------------------------------------

    rth = load_market()

    m5 = resample_m5(rth)

    opening_audit = opening_candle_audit(m5)

    m5 = add_indicators(m5)

    # Only now restrict to analysis period.
    m5 = m5.loc[m5["timestamp_et"] >= ANALYSIS_START].copy()

    m5 = m5.sort_values("timestamp_et").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Fast ATR sweep.
    # -------------------------------------------------------------------------

    result = run_atr_audit(m5)

    print_audit(result)

    # -------------------------------------------------------------------------
    # Baseline ATR14.
    # -------------------------------------------------------------------------

    baseline = result.loc[result["atr_period"].eq(ATR_PERIOD)]

    banner("BASELINE — ATR(14)")

    if baseline.empty:
        print("ATR(14) result not found.")

    else:
        print(
            baseline.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # -------------------------------------------------------------------------
    # Save opening audit.
    # -------------------------------------------------------------------------

    opening_audit.to_csv(
        AUDIT_DIR / "opening_candle_audit.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final audit.
    # -------------------------------------------------------------------------

    banner("FINAL AUDIT")

    print("Opening candle: 09:30–09:35 ET")

    print("Entry: 09:35 ET CLOSE")

    print("ATR: 8 × ATR")

    print(f"ATR periods tested: {ATR_AUDIT_PERIODS}")

    print("HMM/VOL: NOT calculated in this stage")

    print("Robustness: NOT calculated")

    print("Automatic parameter selection: NOT performed")

    print(f"\nResults saved to:\n{AUDIT_DIR}")


if __name__ == "__main__":
    main()
