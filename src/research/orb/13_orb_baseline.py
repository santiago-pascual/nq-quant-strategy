"""
ORB BASELINE — 30m / 2R / 11:00

Purpose
-------
Baseline reproduction of a 30-minute Opening Range Breakout strategy.

Specification
-------------
Instrument: NQ / MNQ
Session: NY RTH
Opening Range: 09:30-10:00 ET
Entry window: 10:00-11:00 ET
Direction:
    LONG  -> break above opening-range high
    SHORT -> break below opening-range low

Execution:
    - 1-minute market data
    - Entry at the breakout level
    - Stop at opposite side of opening range
    - Target = 2R
    - Maximum one trade per session
    - No new entries after 11:00 ET
    - Open position may remain active until SL/TP or RTH close

Important
---------
This is a research baseline, NOT a claim that these are necessarily
the exact author's execution semantics.

No HMM.
No volatility regime.
No optimization.
No parameter selection.

The purpose is to establish the raw ORB edge first.
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


# Canonical project loader
from src.databento_loader import load_databento_mnq


# ============================================================
# CONFIG
# ============================================================

# Session
RTH_START = "09:30"
OR_END = "10:00"
ENTRY_CUTOFF = "11:00"
RTH_END = "16:00"

# ORB
OR_MINUTES = 30
RR = 2.0

# Execution
MAX_TRADES_PER_DAY = 1

# Costs
# Set to zero initially if you want the pure mechanical result.
# Then run again with realistic costs.
COMMISSION_PER_SIDE = 0.0
SLIPPAGE_POINTS = 0.0

# NQ point value.
# 1 NQ point = $20.
# MNQ point value = $2.
#
# R-statistics are independent of this.
NQ_POINT_VALUE = 20.0

# If True, an open position is closed at the final available
# RTH close when neither SL nor TP was hit.
FORCE_EXIT_RTH_CLOSE = True

# Intrabar ambiguity:
#
# If a single 1-minute candle touches both SL and TP,
# we cannot know the true sequence without tick data.
#
# Conservative research convention:
# stop is assumed to happen first.
INTRABAR_BOTH_HIT = "STOP_FIRST"


# ============================================================
# OUTPUT
# ============================================================

RESULT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_FILE = RESULT_DIR / "orb_baseline_trades.csv"
SUMMARY_FILE = RESULT_DIR / "orb_baseline_summary.csv"
DAILY_FILE = RESULT_DIR / "orb_baseline_daily.csv"


# ============================================================
# HELPERS
# ============================================================


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure OHLC columns are numeric.
    """
    out = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def add_session_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NY session date.
    """
    out = df.copy()

    ts = pd.to_datetime(out["timestamp ET"])

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")

    out["timestamp ET"] = ts
    out["session_date"] = ts.dt.date

    return out


def time_mask(
    df: pd.DataFrame,
    start: str,
    end: str,
    include_end: bool = False,
) -> pd.Series:
    """
    Time-of-day mask in NY time.

    For example:
        09:30 <= t < 10:00
    """
    t = df["timestamp ET"].dt.strftime("%H:%M")

    if include_end:
        return (t >= start) & (t <= end)

    return (t >= start) & (t < end)


# ============================================================
# DATA
# ============================================================


def load_market() -> pd.DataFrame:

    print("=" * 80)
    print("LOADING CANONICAL DATABENTO MNQ DATA")
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
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    df = clean_numeric(df)
    df = add_session_date(df)

    df = df.sort_values("timestamp ET").reset_index(drop=True)

    print(f"Rows:  {len(df):,}")
    print(f"Start: {df['timestamp ET'].min()}")
    print(f"End:   {df['timestamp ET'].max()}")

    return df


# ============================================================
# RTH DATA
# ============================================================


def prepare_rth(df: pd.DataFrame) -> pd.DataFrame:

    mask = time_mask(
        df,
        RTH_START,
        RTH_END,
        include_end=False,
    )

    rth = df.loc[mask].copy()

    rth = rth.dropna(subset=["open", "high", "low", "close"])

    rth = rth.sort_values(["session_date", "timestamp ET"])

    rth = rth.reset_index(drop=True)

    return rth


# ============================================================
# OPENING RANGE
# ============================================================


def build_opening_ranges(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    or_df = rth[
        time_mask(
            rth,
            RTH_START,
            OR_END,
            include_end=False,
        )
    ].copy()

    if len(or_df) == 0:
        raise RuntimeError("No opening-range bars found.")

    grouped = (
        or_df.groupby("session_date", sort=True)
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
# TRADE ENGINE
# ============================================================


def execute_session(
    session_df: pd.DataFrame,
    or_row: pd.Series,
) -> dict | None:

    session_date = or_row["session_date"]

    or_high = float(or_row["or_high"])
    or_low = float(or_row["or_low"])
    or_range = float(or_row["or_range"])

    if not np.isfinite(or_range):
        return None

    if or_range <= 0:
        return None

    # --------------------------------------------------------
    # Entry window
    # --------------------------------------------------------

    entry_df = session_df[
        time_mask(
            session_df,
            OR_END,
            ENTRY_CUTOFF,
            include_end=False,
        )
    ].copy()

    if len(entry_df) == 0:
        return None

    entry_df = entry_df.sort_values("timestamp ET")

    # --------------------------------------------------------
    # Search first breakout
    # --------------------------------------------------------

    direction = None
    entry_time = None
    entry_price = None

    for _, bar in entry_df.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        # Both breakouts occur inside same candle.
        #
        # Without tick data we do not know which happened first.
        # Treat this conservatively by skipping the day.
        long_break = high >= or_high
        short_break = low <= or_low

        if long_break and short_break:
            # If opening range is the only information,
            # the sequence is unknowable.
            #
            # Skip rather than introduce directional bias.
            return None

        if long_break:
            direction = "LONG"
            entry_time = bar["timestamp ET"]
            entry_price = or_high
            break

        if short_break:
            direction = "SHORT"
            entry_time = bar["timestamp ET"]
            entry_price = or_low
            break

    if direction is None:
        return None

    # --------------------------------------------------------
    # Execution levels
    # --------------------------------------------------------

    if direction == "LONG":
        stop_price = or_low

        risk_points = entry_price - stop_price

        target_price = entry_price + RR * risk_points

    else:
        stop_price = or_high

        risk_points = stop_price - entry_price

        target_price = entry_price - RR * risk_points

    if risk_points <= 0:
        return None

    # --------------------------------------------------------
    # Manage position
    # --------------------------------------------------------

    after_entry = session_df[session_df["timestamp ET"] > entry_time].copy()

    after_entry = after_entry.sort_values("timestamp ET")

    exit_time = None
    exit_price = None
    exit_reason = None

    for _, bar in after_entry.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        if direction == "LONG":
            hit_stop = low <= stop_price
            hit_target = high >= target_price

            if hit_stop and hit_target:
                if INTRABAR_BOTH_HIT == "STOP_FIRST":
                    exit_price = stop_price
                    exit_reason = "STOP_AND_TARGET_SAME_BAR_STOP_FIRST"
                else:
                    exit_price = target_price
                    exit_reason = "STOP_AND_TARGET_SAME_BAR_TARGET_FIRST"

                exit_time = bar["timestamp ET"]
                break

            if hit_stop:
                exit_price = stop_price
                exit_time = bar["timestamp ET"]
                exit_reason = "INITIAL_STOP"
                break

            if hit_target:
                exit_price = target_price
                exit_time = bar["timestamp ET"]
                exit_reason = "TAKE_PROFIT"
                break

        else:
            hit_stop = high >= stop_price
            hit_target = low <= target_price

            if hit_stop and hit_target:
                if INTRABAR_BOTH_HIT == "STOP_FIRST":
                    exit_price = stop_price
                    exit_reason = "STOP_AND_TARGET_SAME_BAR_STOP_FIRST"
                else:
                    exit_price = target_price
                    exit_reason = "STOP_AND_TARGET_SAME_BAR_TARGET_FIRST"

                exit_time = bar["timestamp ET"]
                break

            if hit_stop:
                exit_price = stop_price
                exit_time = bar["timestamp ET"]
                exit_reason = "INITIAL_STOP"
                break

            if hit_target:
                exit_price = target_price
                exit_time = bar["timestamp ET"]
                exit_reason = "TAKE_PROFIT"
                break

    # --------------------------------------------------------
    # End of session
    # --------------------------------------------------------

    if exit_price is None:
        if FORCE_EXIT_RTH_CLOSE:
            if len(session_df) == 0:
                return None

            last_bar = session_df.iloc[-1]

            exit_time = last_bar["timestamp ET"]
            exit_price = float(last_bar["close"])
            exit_reason = "RTH_CLOSE"

        else:
            return None

    # --------------------------------------------------------
    # R calculation
    # --------------------------------------------------------

    if direction == "LONG":
        raw_points = exit_price - entry_price

    else:
        raw_points = entry_price - exit_price

    # Costs in points
    # Round-trip commission converted into NQ points.
    commission_points = 2.0 * COMMISSION_PER_SIDE / NQ_POINT_VALUE

    total_cost_points = commission_points + 2.0 * SLIPPAGE_POINTS

    net_points = raw_points - total_cost_points

    net_R = net_points / risk_points

    holding_bars = int((exit_time - entry_time).total_seconds() / 60)

    return {
        "session_date": session_date,
        "direction": direction,
        "or_start": or_row["or_start"],
        "or_last": or_row["or_last"],
        "entry_timestamp": entry_time,
        "exit_timestamp": exit_time,
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_range,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_points": risk_points,
        "raw_points": raw_points,
        "net_points": net_points,
        "net_R": net_R,
        "holding_minutes": holding_bars,
        "exit_reason": exit_reason,
        "RR": RR,
    }


# ============================================================
# BACKTEST
# ============================================================


def run_backtest(
    rth: pd.DataFrame,
    opening_ranges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    trades = []

    valid_or = opening_ranges[opening_ranges["valid_or"]].copy()

    valid_dates = set(valid_or["session_date"])

    print()
    print("=" * 80)
    print("RUNNING ORB BASELINE")
    print("=" * 80)

    print(f"Opening Range: {RTH_START} -> {OR_END}")
    print(f"Entry cutoff:  {ENTRY_CUTOFF}")
    print(f"RR:            {RR:.2f}")
    print(f"Max trades/day:{MAX_TRADES_PER_DAY}")

    for session_date, session_df in rth.groupby(
        "session_date",
        sort=True,
    ):
        if session_date not in valid_dates:
            continue

        or_row = valid_or[valid_or["session_date"] == session_date].iloc[0]

        trade = execute_session(
            session_df,
            or_row,
        )

        if trade is not None:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        raise RuntimeError("No ORB trades generated.")

    trades_df = trades_df.sort_values("entry_timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily = trades_df.groupby("session_date", as_index=False).agg(
        daily_R=("net_R", "sum"),
        trades=("net_R", "count"),
    )

    daily["equity_R"] = daily["daily_R"].cumsum()

    daily["peak_R"] = daily["equity_R"].cummax()

    daily["drawdown_R"] = daily["equity_R"] - daily["peak_R"]

    return trades_df, daily


# ============================================================
# METRICS
# ============================================================


def compute_metrics(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict:

    r = trades["net_R"].astype(float)

    n = len(r)

    wins = r[r > 0]
    losses = r[r < 0]

    win_rate = (r > 0).mean() if n else np.nan

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    expectancy = r.mean()

    total_R = r.sum()

    avg_win = wins.mean() if len(wins) else np.nan

    avg_loss = losses.mean() if len(losses) else np.nan

    payoff = (
        avg_win / abs(avg_loss) if np.isfinite(avg_loss) and avg_loss != 0 else np.nan
    )

    max_dd = daily["drawdown_R"].min()

    daily_r = daily["daily_R"]

    daily_std = daily_r.std(ddof=1)

    daily_sharpe = (
        daily_r.mean() / daily_std * math.sqrt(252) if daily_std > 0 else np.nan
    )

    downside = daily_r[daily_r < 0]

    downside_std = downside.std(ddof=1)

    daily_sortino = (
        daily_r.mean() / downside_std * math.sqrt(252) if downside_std > 0 else np.nan
    )

    # --------------------------------------------------------
    # Streaks
    # --------------------------------------------------------

    signs = np.sign(r)

    longest_win = 0
    longest_loss = 0

    current_win = 0
    current_loss = 0

    for s in signs:
        if s > 0:
            current_win += 1
            current_loss = 0

        elif s < 0:
            current_loss += 1
            current_win = 0

        else:
            current_win = 0
            current_loss = 0

        longest_win = max(
            longest_win,
            current_win,
        )

        longest_loss = max(
            longest_loss,
            current_loss,
        )

    # --------------------------------------------------------
    # t-stat
    # --------------------------------------------------------

    r_std = r.std(ddof=1)

    t_stat = expectancy / (r_std / math.sqrt(n)) if r_std > 0 else np.nan

    return {
        "trades": n,
        "sessions": trades["session_date"].nunique(),
        "total_R": total_R,
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
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "mean_holding_minutes": trades["holding_minutes"].mean(),
        "median_holding_minutes": trades["holding_minutes"].median(),
        "long_trades": int((trades["direction"] == "LONG").sum()),
        "short_trades": int((trades["direction"] == "SHORT").sum()),
        "take_profit_exits": int((trades["exit_reason"] == "TAKE_PROFIT").sum()),
        "stop_exits": int(
            trades["exit_reason"]
            .str.contains(
                "STOP",
                na=False,
            )
            .sum()
        ),
        "rth_close_exits": int((trades["exit_reason"] == "RTH_CLOSE").sum()),
        "start_date": trades["session_date"].min(),
        "end_date": trades["session_date"].max(),
    }


# ============================================================
# SIDE ANALYSIS
# ============================================================


def side_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for side in ["LONG", "SHORT"]:
        sub = trades[trades["direction"] == side]

        if len(sub) == 0:
            continue

        r = sub["net_R"]

        wins = r[r > 0]
        losses = r[r < 0]

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        rows.append(
            {
                "direction": side,
                "trades": len(sub),
                "total_R": r.sum(),
                "expectancy_R": r.mean(),
                "win_rate": (r > 0).mean(),
                "profit_factor": pf,
                "avg_win_R": wins.mean() if len(wins) else np.nan,
                "avg_loss_R": losses.mean() if len(losses) else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# YEARLY ANALYSIS
# ============================================================


def yearly_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    tmp = trades.copy()

    tmp["year"] = pd.to_datetime(tmp["session_date"]).dt.year

    out = []

    for year, sub in tmp.groupby(
        "year",
        sort=True,
    ):
        r = sub["net_R"]

        wins = r[r > 0]
        losses = r[r < 0]

        gp = wins.sum()
        gl = abs(losses.sum())

        pf = gp / gl if gl > 0 else np.inf

        out.append(
            {
                "year": year,
                "trades": len(sub),
                "total_R": r.sum(),
                "expectancy_R": r.mean(),
                "win_rate": (r > 0).mean(),
                "profit_factor": pf,
            }
        )

    return pd.DataFrame(out)


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("#" * 80)
    print("# ORB BASELINE — 30m / 2R / 11:00")
    print("#" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    market = load_market()

    # --------------------------------------------------------
    # RTH
    # --------------------------------------------------------

    rth = prepare_rth(market)

    print()
    print(f"RTH rows: {len(rth):,}")

    # --------------------------------------------------------
    # Opening ranges
    # --------------------------------------------------------

    opening_ranges = build_opening_ranges(rth)

    print(f"Sessions: {len(opening_ranges):,}")

    print(
        "Valid 30-bar OR sessions:",
        int(opening_ranges["valid_or"].sum()),
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    trades, daily = run_backtest(
        rth,
        opening_ranges,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_metrics(
        trades,
        daily,
    )

    side_df = side_metrics(trades)

    yearly_df = yearly_metrics(trades)

    summary = pd.DataFrame([metrics])

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    side_file = RESULT_DIR / "orb_baseline_side.csv"

    yearly_file = RESULT_DIR / "orb_baseline_yearly.csv"

    side_df.to_csv(
        side_file,
        index=False,
    )

    yearly_df.to_csv(
        yearly_file,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ORB BASELINE RESULTS")
    print("=" * 80)

    print(f"Trades:              {metrics['trades']:,}")

    print(f"Sessions traded:     {metrics['sessions']:,}")

    print(f"Total R:             {metrics['total_R']:+.4f}R")

    print(f"Expectancy:          {metrics['expectancy_R']:+.6f}R")

    print(f"Win rate:            {metrics['win_rate']:.4%}")

    print(f"Profit factor:       {metrics['profit_factor']:.4f}")

    print(f"Avg win:             {metrics['avg_win_R']:+.4f}R")

    print(f"Avg loss:            {metrics['avg_loss_R']:+.4f}R")

    print(f"Payoff ratio:        {metrics['payoff_ratio']:.4f}")

    print(f"Max DD:              {metrics['max_drawdown_R']:+.4f}R")

    print(f"Daily Sharpe:        {metrics['daily_sharpe']:.4f}")

    print(f"Daily Sortino:       {metrics['daily_sortino']:.4f}")

    print(f"Trade t-stat:        {metrics['trade_t_stat']:.4f}")

    print(f"Longest win streak:  {metrics['longest_win_streak']}")

    print(f"Longest loss streak: {metrics['longest_loss_streak']}")

    print(f"Median holding:      {metrics['median_holding_minutes']:.1f} min")

    print()
    print("--- EXIT BREAKDOWN ---")

    print(f"TP exits:             {metrics['take_profit_exits']:,}")

    print(f"Stop exits:           {metrics['stop_exits']:,}")

    print(f"RTH close exits:      {metrics['rth_close_exits']:,}")

    print()
    print("--- SIDE ---")
    print(side_df.to_string(index=False))

    print()
    print("--- YEARLY ---")
    print(yearly_df.to_string(index=False))

    print()
    print("--- FILES ---")

    print(TRADES_FILE)
    print(DAILY_FILE)
    print(SUMMARY_FILE)
    print(side_file)
    print(yearly_file)

    print()
    print("=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
