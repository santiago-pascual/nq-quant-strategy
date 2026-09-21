"""
ORB EXECUTION AUDIT
===================

Purpose
-------
Audit execution conventions for the ORB baseline.

Fixed strategy parameters
-------------------------
Opening Range: 09:30 -> 10:00 ET
Entry cutoff:  11:00 ET
RR:            2.0
Max trades:    1 per session

Execution variants
------------------
1. TOUCH
   Enter at the OR boundary when price breaks it.

2. CLOSE
   Enter at the close of the first 1-minute candle that
   closes outside the Opening Range.

3. NEXT_OPEN
   Confirm the breakout on a 1-minute close and enter
   at the next 1-minute bar open.

For TOUCH, two same-bar management variants are tested:

4. TOUCH_NEXT_BAR
   Entry occurs at the breakout level, but SL/TP management
   starts on the following 1-minute bar.

5. TOUCH_SAME_BAR_STOP_FIRST
   Entry occurs at the breakout level and the remainder of
   the breakout candle is managed immediately.
   If SL and TP are both touched in the same candle,
   STOP is assumed first.

Important
---------
This is an execution audit, NOT parameter optimization.

Do not select a variant simply because it has the highest PF.
The purpose is to understand how much of the result depends
on execution convention.
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.databento_loader import load_databento_mnq


# ============================================================
# CONFIG
# ============================================================

RTH_START = "09:30"
OR_END = "10:00"
ENTRY_CUTOFF = "11:00"
RTH_END = "16:00"

OR_MINUTES = 30
RR = 2.0

FORCE_EXIT_RTH_CLOSE = True

# Same-bar ambiguity convention
SAME_BAR_POLICY = "STOP_FIRST"


# ============================================================
# OUTPUT
# ============================================================

RESULT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRADES_FILE = RESULT_DIR / "orb_execution_audit_trades.csv"

SUMMARY_FILE = RESULT_DIR / "orb_execution_audit_summary.csv"

YEARLY_FILE = RESULT_DIR / "orb_execution_audit_yearly.csv"

SIDE_FILE = RESULT_DIR / "orb_execution_audit_side.csv"


# ============================================================
# DATA HELPERS
# ============================================================


def load_market() -> pd.DataFrame:

    print("=" * 80)
    print("LOADING CANONICAL DATABENTO DATA")
    print("=" * 80)

    df = load_databento_mnq()

    if df is None or len(df) == 0:
        raise RuntimeError("Canonical Databento loader returned no data.")

    required = {
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    ts = pd.to_datetime(df["timestamp ET"])

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")

    df["timestamp ET"] = ts

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["session_date"] = df["timestamp ET"].dt.date

    df = df.sort_values("timestamp ET").reset_index(drop=True)

    print(f"Rows:  {len(df):,}")

    print(f"Start: {df['timestamp ET'].min()}")

    print(f"End:   {df['timestamp ET'].max()}")

    return df


def prepare_rth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    t = df["timestamp ET"].dt.strftime("%H:%M")

    mask = (t >= RTH_START) & (t < RTH_END)

    rth = df.loc[mask].copy()

    rth = rth.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    rth = rth.sort_values(
        [
            "session_date",
            "timestamp ET",
        ]
    )

    rth = rth.reset_index(drop=True)

    return rth


def build_opening_ranges(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    t = rth["timestamp ET"].dt.strftime("%H:%M")

    mask = (t >= RTH_START) & (t < OR_END)

    opening = rth.loc[mask].copy()

    grouped = (
        opening.groupby(
            "session_date",
            sort=True,
        )
        .agg(
            or_high=("high", "max"),
            or_low=("low", "min"),
            or_open=("open", "first"),
            or_close=("close", "last"),
            or_bars=("timestamp ET", "count"),
            or_start=("timestamp ET", "min"),
            or_last=("timestamp ET", "max"),
        )
        .reset_index()
    )

    grouped["or_range"] = grouped["or_high"] - grouped["or_low"]

    grouped["valid_or"] = grouped["or_bars"] == OR_MINUTES

    return grouped


# ============================================================
# LEVELS
# ============================================================


def make_levels(
    direction: str,
    entry_price: float,
    or_high: float,
    or_low: float,
) -> tuple[float, float, float]:

    if direction == "LONG":
        stop = or_low

        risk = entry_price - stop

        target = entry_price + RR * risk

    else:
        stop = or_high

        risk = stop - entry_price

        target = entry_price - RR * risk

    return stop, target, risk


# ============================================================
# EXIT ENGINE
# ============================================================


def manage_position(
    bars: pd.DataFrame,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> tuple:

    for _, bar in bars.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        timestamp = bar["timestamp ET"]

        if direction == "LONG":
            hit_stop = low <= stop_price

            hit_target = high >= target_price

        else:
            hit_stop = high >= stop_price

            hit_target = low <= target_price

        # ----------------------------------------------------
        # Both levels touched
        # ----------------------------------------------------

        if hit_stop and hit_target:
            if SAME_BAR_POLICY == "STOP_FIRST":
                return (
                    timestamp,
                    stop_price,
                    "STOP_AND_TARGET_SAME_BAR_STOP_FIRST",
                )

            return (
                timestamp,
                target_price,
                "STOP_AND_TARGET_SAME_BAR_TARGET_FIRST",
            )

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        if hit_stop:
            return (
                timestamp,
                stop_price,
                "INITIAL_STOP",
            )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        if hit_target:
            return (
                timestamp,
                target_price,
                "TAKE_PROFIT",
            )

    # --------------------------------------------------------
    # RTH close
    # --------------------------------------------------------

    if FORCE_EXIT_RTH_CLOSE and len(bars) > 0:
        last = bars.iloc[-1]

        return (
            last["timestamp ET"],
            float(last["close"]),
            "RTH_CLOSE",
        )

    return (
        None,
        None,
        None,
    )


# ============================================================
# SINGLE SESSION — TOUCH
# ============================================================


def run_touch(
    session_df: pd.DataFrame,
    or_row: pd.Series,
    same_bar_management: bool,
) -> dict | None:

    or_high = float(or_row["or_high"])
    or_low = float(or_row["or_low"])

    t = session_df["timestamp ET"].dt.strftime("%H:%M")

    entry_window = session_df[(t >= OR_END) & (t < ENTRY_CUTOFF)].copy()

    entry_window = entry_window.sort_values("timestamp ET")

    for idx, (_, bar) in enumerate(entry_window.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        long_break = high >= or_high
        short_break = low <= or_low

        # Both sides touched during same minute.
        # Sequence is unknowable.
        if long_break and short_break:
            return None

        if not long_break and not short_break:
            continue

        if long_break:
            direction = "LONG"
            entry_price = or_high

        else:
            direction = "SHORT"
            entry_price = or_low

        stop, target, risk = make_levels(
            direction,
            entry_price,
            or_high,
            or_low,
        )

        if risk <= 0:
            return None

        # ----------------------------------------------------
        # Which bars are allowed to manage the trade?
        # ----------------------------------------------------

        if same_bar_management:
            # Include breakout candle.
            manage = entry_window.iloc[idx:].copy()

        else:
            # Start from next candle.
            manage = entry_window.iloc[idx + 1 :].copy()

        # Add all remaining RTH bars after the
        # entry window if necessary.
        entry_timestamp = bar["timestamp ET"]

        later_rth = session_df[session_df["timestamp ET"] > entry_timestamp].copy()

        if same_bar_management:
            # Breakout candle already included.
            later = later_rth[later_rth["timestamp ET"] > entry_timestamp]

            manage = pd.concat(
                [
                    manage.iloc[:1],
                    later,
                ],
                ignore_index=True,
            )

        else:
            manage = later_rth.copy()

        exit_time, exit_price, reason = manage_position(
            manage,
            direction,
            entry_price,
            stop,
            target,
        )

        if exit_price is None:
            return None

        if direction == "LONG":
            raw_points = exit_price - entry_price
        else:
            raw_points = entry_price - exit_price

        net_R = raw_points / risk

        return {
            "session_date": or_row["session_date"],
            "execution": (
                "TOUCH_SAME_BAR" if same_bar_management else "TOUCH_NEXT_BAR"
            ),
            "direction": direction,
            "entry_timestamp": entry_timestamp,
            "exit_timestamp": exit_time,
            "or_high": or_high,
            "or_low": or_low,
            "or_range": float(or_row["or_range"]),
            "entry_price": entry_price,
            "stop_price": stop,
            "target_price": target,
            "risk_points": risk,
            "raw_points": raw_points,
            "net_R": net_R,
            "exit_reason": reason,
            "holding_minutes": (exit_time - entry_timestamp).total_seconds() / 60.0,
        }

    return None


# ============================================================
# SINGLE SESSION — CLOSE
# ============================================================


def run_close(
    session_df: pd.DataFrame,
    or_row: pd.Series,
    next_open: bool,
) -> dict | None:

    or_high = float(or_row["or_high"])
    or_low = float(or_row["or_low"])

    t = session_df["timestamp ET"].dt.strftime("%H:%M")

    entry_window = session_df[(t >= OR_END) & (t < ENTRY_CUTOFF)].copy()

    entry_window = entry_window.sort_values("timestamp ET").reset_index(drop=True)

    for i in range(len(entry_window)):
        bar = entry_window.iloc[i]

        close = float(bar["close"])

        # ----------------------------------------------------
        # LONG close breakout
        # ----------------------------------------------------

        if close > or_high:
            direction = "LONG"

        # ----------------------------------------------------
        # SHORT close breakout
        # ----------------------------------------------------

        elif close < or_low:
            direction = "SHORT"

        else:
            continue

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        if next_open:
            if i + 1 >= len(entry_window):
                return None

            next_bar = entry_window.iloc[i + 1]

            entry_timestamp = next_bar["timestamp ET"]

            # Entry must still occur before cutoff.
            if entry_timestamp >= pd.Timestamp(
                f"{entry_timestamp.date()} {ENTRY_CUTOFF}",
                tz=entry_timestamp.tz,
            ):
                return None

            entry_price = float(next_bar["open"])

            # Manage starting on entry candle.
            manage = session_df[session_df["timestamp ET"] >= entry_timestamp].copy()

            execution_name = "NEXT_OPEN"

        else:
            entry_timestamp = bar["timestamp ET"]

            entry_price = close

            # Entry occurs at candle close.
            # Therefore management starts next bar.
            manage = session_df[session_df["timestamp ET"] > entry_timestamp].copy()

            execution_name = "CLOSE"

        stop, target, risk = make_levels(
            direction,
            entry_price,
            or_high,
            or_low,
        )

        if risk <= 0:
            return None

        exit_time, exit_price, reason = manage_position(
            manage,
            direction,
            entry_price,
            stop,
            target,
        )

        if exit_price is None:
            return None

        if direction == "LONG":
            raw_points = exit_price - entry_price
        else:
            raw_points = entry_price - exit_price

        net_R = raw_points / risk

        return {
            "session_date": or_row["session_date"],
            "execution": execution_name,
            "direction": direction,
            "entry_timestamp": entry_timestamp,
            "exit_timestamp": exit_time,
            "or_high": or_high,
            "or_low": or_low,
            "or_range": float(or_row["or_range"]),
            "entry_price": entry_price,
            "stop_price": stop,
            "target_price": target,
            "risk_points": risk,
            "raw_points": raw_points,
            "net_R": net_R,
            "exit_reason": reason,
            "holding_minutes": (exit_time - entry_timestamp).total_seconds() / 60.0,
        }

    return None


# ============================================================
# BACKTEST ONE VARIANT
# ============================================================


def run_variant(
    rth: pd.DataFrame,
    opening_ranges: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:

    valid = opening_ranges[opening_ranges["valid_or"]].copy()

    valid_map = {row["session_date"]: row for _, row in valid.iterrows()}

    trades = []

    for session_date, session_df in rth.groupby(
        "session_date",
        sort=True,
    ):
        if session_date not in valid_map:
            continue

        or_row = valid_map[session_date]

        if variant == "TOUCH_NEXT_BAR":
            trade = run_touch(
                session_df,
                or_row,
                same_bar_management=False,
            )

        elif variant == "TOUCH_SAME_BAR":
            trade = run_touch(
                session_df,
                or_row,
                same_bar_management=True,
            )

        elif variant == "CLOSE":
            trade = run_close(
                session_df,
                or_row,
                next_open=False,
            )

        elif variant == "NEXT_OPEN":
            trade = run_close(
                session_df,
                or_row,
                next_open=True,
            )

        else:
            raise ValueError(f"Unknown variant: {variant}")

        if trade is not None:
            trades.append(trade)

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    return df


# ============================================================
# METRICS
# ============================================================


def metrics(
    trades: pd.DataFrame,
) -> dict:

    r = trades["net_R"].astype(float)

    n = len(r)

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    expectancy = r.mean()

    win_rate = (r > 0).mean()

    avg_win = wins.mean() if len(wins) else np.nan

    avg_loss = losses.mean() if len(losses) else np.nan

    payoff = avg_win / abs(avg_loss) if len(losses) else np.nan

    # --------------------------------------------------------
    # Daily
    # --------------------------------------------------------

    daily = trades.groupby("session_date")["net_R"].sum()

    equity = daily.cumsum()

    drawdown = equity - equity.cummax()

    max_dd = drawdown.min()

    daily_std = daily.std(ddof=1)

    daily_sharpe = (
        daily.mean() / daily_std * math.sqrt(252) if daily_std > 0 else np.nan
    )

    downside = daily[daily < 0]

    downside_std = downside.std(ddof=1)

    daily_sortino = (
        daily.mean() / downside_std * math.sqrt(252) if downside_std > 0 else np.nan
    )

    # --------------------------------------------------------
    # t-stat
    # --------------------------------------------------------

    std = r.std(ddof=1)

    t_stat = expectancy / (std / math.sqrt(n)) if std > 0 else np.nan

    # --------------------------------------------------------
    # Streaks
    # --------------------------------------------------------

    longest_win = 0
    longest_loss = 0

    cw = 0
    cl = 0

    for value in r:
        if value > 0:
            cw += 1
            cl = 0

        elif value < 0:
            cl += 1
            cw = 0

        else:
            cw = 0
            cl = 0

        longest_win = max(
            longest_win,
            cw,
        )

        longest_loss = max(
            longest_loss,
            cl,
        )

    return {
        "execution": trades["execution"].iloc[0],
        "trades": n,
        "total_R": r.sum(),
        "expectancy_R": expectancy,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "payoff_ratio": payoff,
        "max_drawdown_R": max_dd,
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "trade_t_stat": t_stat,
        "long_trades": int((trades["direction"] == "LONG").sum()),
        "short_trades": int((trades["direction"] == "SHORT").sum()),
        "TP_exits": int((trades["exit_reason"] == "TAKE_PROFIT").sum()),
        "stop_exits": int(
            trades["exit_reason"]
            .str.contains(
                "STOP",
                na=False,
            )
            .sum()
        ),
        "RTH_close_exits": int((trades["exit_reason"] == "RTH_CLOSE").sum()),
        "median_holding_minutes": trades["holding_minutes"].median(),
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "start_date": trades["session_date"].min(),
        "end_date": trades["session_date"].max(),
    }


# ============================================================
# YEARLY
# ============================================================


def yearly_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    tmp = trades.copy()

    tmp["year"] = pd.to_datetime(tmp["session_date"]).dt.year

    rows = []

    for (execution, year), sub in tmp.groupby(
        ["execution", "year"],
        sort=True,
    ):
        r = sub["net_R"]

        wins = r[r > 0]
        losses = r[r < 0]

        gp = wins.sum()
        gl = abs(losses.sum())

        pf = gp / gl if gl > 0 else np.inf

        rows.append(
            {
                "execution": execution,
                "year": year,
                "trades": len(sub),
                "total_R": r.sum(),
                "expectancy_R": r.mean(),
                "win_rate": (r > 0).mean(),
                "profit_factor": pf,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SIDE
# ============================================================


def side_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (execution, side), sub in trades.groupby(
        [
            "execution",
            "direction",
        ],
        sort=True,
    ):
        r = sub["net_R"]

        wins = r[r > 0]
        losses = r[r < 0]

        gp = wins.sum()
        gl = abs(losses.sum())

        pf = gp / gl if gl > 0 else np.inf

        rows.append(
            {
                "execution": execution,
                "direction": side,
                "trades": len(sub),
                "total_R": r.sum(),
                "expectancy_R": r.mean(),
                "win_rate": (r > 0).mean(),
                "profit_factor": pf,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("#" * 80)
    print("# ORB EXECUTION AUDIT")
    print("#" * 80)

    market = load_market()

    rth = prepare_rth(market)

    opening_ranges = build_opening_ranges(rth)

    print()
    print(f"RTH rows: {len(rth):,}")

    print(f"Sessions: {len(opening_ranges):,}")

    print(
        "Valid 30m OR sessions:",
        int(opening_ranges["valid_or"].sum()),
    )

    variants = [
        "TOUCH_NEXT_BAR",
        "TOUCH_SAME_BAR",
        "CLOSE",
        "NEXT_OPEN",
    ]

    all_trades = []
    summaries = []

    for variant in variants:
        print()
        print("-" * 80)
        print(f"RUNNING: {variant}")
        print("-" * 80)

        trades = run_variant(
            rth,
            opening_ranges,
            variant,
        )

        if len(trades) == 0:
            print("No trades.")

            continue

        all_trades.append(trades)

        m = metrics(trades)

        summaries.append(m)

        print(f"Trades:       {m['trades']:,}")

        print(f"Total R:      {m['total_R']:+.4f}R")

        print(f"Expectancy:   {m['expectancy_R']:+.6f}R")

        print(f"Win rate:     {m['win_rate']:.4%}")

        print(f"PF:           {m['profit_factor']:.4f}")

        print(f"Max DD:       {m['max_drawdown_R']:+.4f}R")

        print(f"Sharpe:       {m['daily_sharpe']:.4f}")

        print(f"Sortino:      {m['daily_sortino']:.4f}")

        print(f"t-stat:       {m['trade_t_stat']:.4f}")

        print(f"LONG:         {m['long_trades']:,}")

        print(f"SHORT:        {m['short_trades']:,}")

        print(f"TP exits:     {m['TP_exits']:,}")

        print(f"Stop exits:   {m['stop_exits']:,}")

        print(f"RTH exits:    {m['RTH_close_exits']:,}")

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if not all_trades:
        raise RuntimeError("No trades produced.")

    trades_all = pd.concat(
        all_trades,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summaries)

    yearly_df = yearly_metrics(trades_all)

    side_df = side_metrics(trades_all)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trades_all.to_csv(
        TRADES_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    yearly_df.to_csv(
        YEARLY_FILE,
        index=False,
    )

    side_df.to_csv(
        SIDE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EXECUTION AUDIT SUMMARY")
    print("=" * 80)

    cols = [
        "execution",
        "trades",
        "total_R",
        "expectancy_R",
        "win_rate",
        "profit_factor",
        "max_drawdown_R",
        "daily_sharpe",
        "daily_sortino",
        "trade_t_stat",
    ]

    print(summary_df[cols].to_string(index=False))

    print()
    print("=" * 80)
    print("YEARLY RESULTS")
    print("=" * 80)

    print(yearly_df.to_string(index=False))

    print()
    print("=" * 80)
    print("SIDE RESULTS")
    print("=" * 80)

    print(side_df.to_string(index=False))

    print()
    print("=" * 80)
    print("FILES")
    print("=" * 80)

    print(TRADES_FILE)
    print(SUMMARY_FILE)
    print(YEARLY_FILE)
    print(SIDE_FILE)

    print()
    print("=" * 80)
    print("EXECUTION AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
