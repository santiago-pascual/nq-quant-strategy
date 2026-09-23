"""
13D — SESSION MOMENTUM: EMA120 TRAILING MECHANICS AUDIT

Purpose
-------
Isolate the EMA120 trailing-stop implementation.

Everything else is fixed:

    M5 RTH
    opening candle = 09:30 -> 09:35 ET
    signal = opening close vs EMA12
    entry = 09:35 ET opening-candle CLOSE
    ATR = 14
    initial stop = 8 * ATR
    activation = +0.5R
    one trade per session
    no fixed target
    positions may carry overnight

Only the trailing-stop mechanics change.

Variants
--------
A:
    Current baseline.
    EMA120 from current bar.
    Once +0.5R is touched, current-bar EMA120 can become trail.

B:
    EMA120 from previous completed M5 bar.

C:
    Activation is detected on the current bar, but EMA120 trail
    becomes effective only from the NEXT bar.

D:
    The activation bar cannot simultaneously activate and execute
    the EMA trail. Trail starts on the following bar.

E:
    Conservative combination:
    previous-bar EMA120 + next-bar activation.

No HMM/VOL.
No parameter optimization.
No Monte Carlo.
No robustness selection.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATH
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# IMPORTS
# =============================================================================

from src.databento_loader import load_databento_mnq
from src.session_engine import add_session_information


# =============================================================================
# CONFIG
# =============================================================================

EMA_SIGNAL = 12
EMA_TRAIL = 120

ATR_PERIOD = 14
ATR_MULTIPLIER = 8.0

TRAIL_ACTIVATION_R = 0.5

NY_OPEN_HOUR = 9
NY_OPEN_MINUTE = 30

ANALYSIS_START = pd.Timestamp(
    "2020-01-01",
    tz="America/New_York",
)

IS_END = pd.Timestamp(
    "2024-12-31 23:59:59",
    tz="America/New_York",
)


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
# VARIANTS
# =============================================================================

VARIANTS = {
    "A_baseline": {
        "description": (
            "Current-bar EMA120; activation bar may immediately set trail."
        ),
    },
    "B_previous_ema": {
        "description": ("Previous completed bar EMA120."),
    },
    "C_next_bar_activation": {
        "description": ("Activation detected now; trail begins next bar."),
    },
    "D_no_same_bar_trail": {
        "description": ("Activation bar cannot execute the new EMA trail."),
    },
    "E_conservative": {
        "description": ("Previous-bar EMA120 + next-bar trail activation."),
    },
}


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


def max_drawdown(
    equity: pd.Series,
) -> float:

    if equity.empty:
        return np.nan

    peak = equity.cummax()

    return float((equity - peak).min())


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


# =============================================================================
# MARKET
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

    df = add_session_information(df)

    if "market_period" in df.columns:
        df = df.loc[df["market_period"].eq("RTH")].copy()

    t = df["timestamp_et"].dt.time

    start = pd.Timestamp("09:30").time()

    end = pd.Timestamp("16:00").time()

    df = df.loc[(t >= start) & (t < end)].copy()

    return df


# =============================================================================
# M5
# =============================================================================


def build_m5(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    banner("2. BUILD RTH M5")

    x = rth.copy().set_index("timestamp_et")

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

    m5 = m5.loc[m5["minute_count"].eq(5)].copy()

    m5 = m5.drop(columns=["minute_count"])

    m5["session_date"] = m5["timestamp_et"].dt.date

    m5 = m5.sort_values("timestamp_et").reset_index(drop=True)

    print(f"Complete M5 bars: {len(m5):,}")

    print(f"Sessions: {m5['session_date'].nunique():,}")

    return m5


# =============================================================================
# INDICATORS
# =============================================================================


def add_indicators(
    m5: pd.DataFrame,
) -> pd.DataFrame:

    banner("3. BUILD EMA12 / EMA120 / ATR14")

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

    x["atr"] = (
        x["true_range"]
        .ewm(
            alpha=1.0 / ATR_PERIOD,
            adjust=False,
            min_periods=ATR_PERIOD,
        )
        .mean()
    )

    # Previous completed-bar EMA.
    x["ema120_previous"] = x["ema120"].shift(1)

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

    activation_time: pd.Timestamp | None = None

    activation_bar_index: int | None = None

    trail_stop: float | None = None


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
# STRATEGY ENGINE
# =============================================================================


def run_strategy(
    m5: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:

    x = m5.copy().sort_values("timestamp_et").reset_index(drop=True)

    trades: list[dict] = []

    pos: Position | None = None

    sessions_with_entry: set = set()

    for i, row in x.iterrows():
        ts = row["timestamp_et"]

        # =====================================================================
        # MANAGE OPEN POSITION
        # =====================================================================

        if pos is not None:
            # -----------------------------------------------------------------
            # LONG
            # -----------------------------------------------------------------

            if pos.side == "LONG":
                # =============================================================
                # Determine stop available at START of this bar.
                # =============================================================

                if pos.activated and pos.trail_stop is not None:
                    current_stop = pos.trail_stop

                else:
                    current_stop = pos.initial_stop

                # =============================================================
                # Stop check FIRST.
                # =============================================================

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

                # =============================================================
                # Activation threshold.
                # =============================================================

                activation_level = (
                    pos.entry_price + TRAIL_ACTIVATION_R * pos.risk_points
                )

                hit_activation = not pos.activated and row["high"] >= activation_level

                if hit_activation:
                    pos.activated = True

                    pos.activation_time = ts

                    pos.activation_bar_index = i

                    # ---------------------------------------------------------
                    # A / B:
                    # Same-bar trail can be established.
                    # ---------------------------------------------------------

                    if variant == "A_baseline":
                        pos.trail_stop = max(
                            pos.initial_stop,
                            float(row["ema120"]),
                        )

                    elif variant == "B_previous_ema":
                        ema = row["ema120_previous"]

                        if np.isfinite(ema):
                            pos.trail_stop = max(
                                pos.initial_stop,
                                float(ema),
                            )
                        else:
                            pos.trail_stop = pos.initial_stop

                    # ---------------------------------------------------------
                    # C / D / E:
                    # Do NOT create usable trail yet.
                    # It becomes effective next bar.
                    # ---------------------------------------------------------

                    else:
                        pos.trail_stop = None

                elif pos.activated:
                    # ---------------------------------------------------------
                    # C / D:
                    # Current EMA, but only after activation bar.
                    # ---------------------------------------------------------

                    if variant in {
                        "C_next_bar_activation",
                        "D_no_same_bar_trail",
                    }:
                        if (
                            pos.activation_bar_index is not None
                            and i > pos.activation_bar_index
                        ):
                            ema = float(row["ema120"])

                            if pos.trail_stop is None:
                                pos.trail_stop = max(
                                    pos.initial_stop,
                                    ema,
                                )

                            else:
                                pos.trail_stop = max(
                                    float(pos.trail_stop),
                                    ema,
                                )

                    # ---------------------------------------------------------
                    # E:
                    # Previous-bar EMA.
                    # ---------------------------------------------------------

                    elif variant == "E_conservative":
                        ema = row["ema120_previous"]

                        if np.isfinite(ema):
                            if pos.trail_stop is None:
                                pos.trail_stop = max(
                                    pos.initial_stop,
                                    float(ema),
                                )

                            else:
                                pos.trail_stop = max(
                                    float(pos.trail_stop),
                                    float(ema),
                                )

            # -----------------------------------------------------------------
            # SHORT
            # -----------------------------------------------------------------

            else:
                if pos.activated and pos.trail_stop is not None:
                    current_stop = pos.trail_stop

                else:
                    current_stop = pos.initial_stop

                # =============================================================
                # Stop check.
                # =============================================================

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

                # =============================================================
                # Activation.
                # =============================================================

                activation_level = (
                    pos.entry_price - TRAIL_ACTIVATION_R * pos.risk_points
                )

                hit_activation = not pos.activated and row["low"] <= activation_level

                if hit_activation:
                    pos.activated = True

                    pos.activation_time = ts

                    pos.activation_bar_index = i

                    if variant == "A_baseline":
                        pos.trail_stop = min(
                            pos.initial_stop,
                            float(row["ema120"]),
                        )

                    elif variant == "B_previous_ema":
                        ema = row["ema120_previous"]

                        if np.isfinite(ema):
                            pos.trail_stop = min(
                                pos.initial_stop,
                                float(ema),
                            )

                        else:
                            pos.trail_stop = pos.initial_stop

                    else:
                        pos.trail_stop = None

                elif pos.activated:
                    if variant in {
                        "C_next_bar_activation",
                        "D_no_same_bar_trail",
                    }:
                        if (
                            pos.activation_bar_index is not None
                            and i > pos.activation_bar_index
                        ):
                            ema = float(row["ema120"])

                            if pos.trail_stop is None:
                                pos.trail_stop = min(
                                    pos.initial_stop,
                                    ema,
                                )

                            else:
                                pos.trail_stop = min(
                                    float(pos.trail_stop),
                                    ema,
                                )

                    elif variant == "E_conservative":
                        ema = row["ema120_previous"]

                        if np.isfinite(ema):
                            if pos.trail_stop is None:
                                pos.trail_stop = min(
                                    pos.initial_stop,
                                    float(ema),
                                )

                            else:
                                pos.trail_stop = min(
                                    float(pos.trail_stop),
                                    float(ema),
                                )

            continue

        # =====================================================================
        # NO POSITION — LOOK FOR OPENING SIGNAL
        # =====================================================================

        if ts.hour != NY_OPEN_HOUR or ts.minute != NY_OPEN_MINUTE:
            continue

        session = row["session_date"]

        if session in sessions_with_entry:
            continue

        if (
            not np.isfinite(row["ema12"])
            or not np.isfinite(row["ema120"])
            or not np.isfinite(row["atr"])
            or row["atr"] <= 0
        ):
            continue

        # =====================================================================
        # SIGNAL
        # =====================================================================

        if row["close"] > row["ema12"]:
            side = "LONG"

        elif row["close"] < row["ema12"]:
            side = "SHORT"

        else:
            continue

        # =====================================================================
        # ENTRY = 09:35 CLOSE
        # =====================================================================

        entry_price = float(row["close"])

        entry_time = ts + pd.Timedelta(minutes=5)

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

    # =========================================================================
    # END OF DATA
    # =========================================================================

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


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict:

    if trades.empty:
        return {
            "trades": 0,
            "total_R": np.nan,
            "expectancy_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "t_stat": np.nan,
            "daily_sharpe": np.nan,
        }

    r = trades["r"].astype(float)

    return {
        "trades": len(r),
        "total_R": float(r.sum()),
        "expectancy_R": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": (profit_factor(r)),
        "max_drawdown_R": (max_drawdown(r.cumsum())),
        "t_stat": t_stat(r),
        "daily_sharpe": (daily_sharpe(trades)),
    }


# =============================================================================
# RUN AUDIT
# =============================================================================


def run_audit(
    m5: pd.DataFrame,
) -> pd.DataFrame:

    banner("4. EMA120 TRAILING MECHANICS AUDIT")

    rows = []

    total = len(VARIANTS)

    for n, (
        variant,
        config,
    ) in enumerate(
        VARIANTS.items(),
        start=1,
    ):
        print(
            f"\n[{n}/{total}] {variant}",
            flush=True,
        )

        print(f"Description: {config['description']}")

        start = time.perf_counter()

        trades = run_strategy(
            m5,
            variant,
        )

        if trades.empty:
            print("ZERO TRADES")

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

        full = calculate_metrics(trades)

        is_metrics = calculate_metrics(trades.loc[trades["sample"].eq("IS")])

        oos_metrics = calculate_metrics(trades.loc[trades["sample"].eq("OOS")])

        elapsed = time.perf_counter() - start

        rows.append(
            {
                "variant": variant,
                "description": config["description"],
                "trades": full["trades"],
                "total_R": full["total_R"],
                "expectancy_R": full["expectancy_R"],
                "win_rate": full["win_rate"],
                "profit_factor": full["profit_factor"],
                "max_drawdown_R": full["max_drawdown_R"],
                "daily_sharpe": full["daily_sharpe"],
                "t_stat": full["t_stat"],
                "is_trades": is_metrics["trades"],
                "is_total_R": is_metrics["total_R"],
                "is_expectancy_R": is_metrics["expectancy_R"],
                "is_profit_factor": is_metrics["profit_factor"],
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
        )

        trades.to_csv(
            AUDIT_DIR / f"trailing_{variant}_trades.csv",
            index=False,
        )

        print(
            f"[DONE] {variant} | "
            f"Trades={len(trades):,} | "
            f"PF={full['profit_factor']:.4f} | "
            f"Exp={full['expectancy_R']:.5f}R | "
            f"OOS PF={oos_metrics['profit_factor']:.4f} | "
            f"OOS Exp={oos_metrics['expectancy_R']:.5f}R | "
            f"{elapsed:.2f}s",
            flush=True,
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        AUDIT_DIR / "trailing_mechanics_audit.csv",
        index=False,
    )

    return result


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("SESSION MOMENTUM — EMA120 TRAILING AUDIT")

    print(f"ATR period:    {ATR_PERIOD}")

    print(f"ATR multiplier: {ATR_MULTIPLIER}")

    print(f"EMA signal:    {EMA_SIGNAL}")

    print(f"EMA trail:     {EMA_TRAIL}")

    print(f"Trail trigger: +{TRAIL_ACTIVATION_R}R")

    print("Opening candle: 09:30–09:35 ET")

    print("Entry:          09:35 ET CLOSE")

    print("\nOnly EMA120 trailing mechanics vary.")

    # -------------------------------------------------------------------------
    # Load once.
    # -------------------------------------------------------------------------

    rth = load_market()

    m5 = build_m5(rth)

    m5 = add_indicators(m5)

    # -------------------------------------------------------------------------
    # Analysis period.
    # -------------------------------------------------------------------------

    m5 = m5.loc[m5["timestamp_et"] >= ANALYSIS_START].copy()

    m5 = m5.sort_values("timestamp_et").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Audit.
    # -------------------------------------------------------------------------

    result = run_audit(m5)

    # -------------------------------------------------------------------------
    # Output.
    # -------------------------------------------------------------------------

    banner("TRAILING AUDIT — FULL SAMPLE")

    print(
        result[
            [
                "variant",
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

    banner("TRAILING AUDIT — OOS")

    print(
        result[
            [
                "variant",
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

    banner("FINAL AUDIT")

    print("ATR: 14")

    print("ATR multiplier: 8")

    print("Opening candle: 09:30–09:35 ET")

    print("Entry: 09:35 ET CLOSE")

    print("Only trailing mechanics varied.")

    print("\nSaved:")

    print(AUDIT_DIR / "trailing_mechanics_audit.csv")


if __name__ == "__main__":
    main()
