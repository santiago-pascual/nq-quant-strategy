"""
ORB RECONCILIATION TEST
=======================

Purpose
-------
Compare the ORB 30m / 2R / 11:00 baseline against the exact
OOS period shown in the external study.

External-study OOS period:
    2020-06-23 -> 2026-06-19

Strategy:
    Opening Range: 09:30-10:00 ET
    Breakout: touch OR high/low
    RR: 2.0
    Entry cutoff: 11:00 ET
    One trade per session
    Exit: SL / TP / RTH close

This script does NOT optimize parameters.

It reports:
    1. Full dataset
    2. Exact external OOS period
    3. Year-by-year results inside the external OOS
    4. LONG / SHORT
    5. Trade counts and date coverage

Metrics are expressed in R, so contract sizing does not affect
the strategy comparison.
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

# External study period shown in the screenshot
EXTERNAL_OOS_START = pd.Timestamp("2020-06-23").date()

EXTERNAL_OOS_END = pd.Timestamp("2026-06-19").date()

# Conservative handling if both sides of the OR are touched
# in the same 1-minute candle.
SKIP_BOTH_BREAKS = True

# If no SL/TP is hit, close at the final RTH bar.
FORCE_EXIT_RTH_CLOSE = True


# ============================================================
# OUTPUT
# ============================================================

RESULT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRADES_FILE = RESULT_DIR / "orb_reconciliation_trades.csv"

SUMMARY_FILE = RESULT_DIR / "orb_reconciliation_summary.csv"

YEARLY_FILE = RESULT_DIR / "orb_reconciliation_yearly.csv"

SIDE_FILE = RESULT_DIR / "orb_reconciliation_side.csv"


# ============================================================
# LOAD DATA
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
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

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


# ============================================================
# PREPARE RTH
# ============================================================


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


# ============================================================
# OPENING RANGE
# ============================================================


def build_opening_ranges(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    t = rth["timestamp ET"].dt.strftime("%H:%M")

    opening = rth[(t >= RTH_START) & (t < OR_END)].copy()

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
# METRICS
# ============================================================


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict:

    if len(trades) == 0:
        return {
            "trades": 0,
            "sessions": 0,
            "total_R": np.nan,
            "expectancy_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "daily_sharpe": np.nan,
            "daily_sortino": np.nan,
            "trade_t_stat": np.nan,
            "avg_win_R": np.nan,
            "avg_loss_R": np.nan,
            "payoff_ratio": np.nan,
            "long_trades": 0,
            "short_trades": 0,
            "TP_exits": 0,
            "stop_exits": 0,
            "RTH_close_exits": 0,
        }

    r = trades["net_R"].astype(float)

    n = len(r)

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    expectancy = r.mean()

    win_rate = r.gt(0).mean()

    avg_win = wins.mean() if len(wins) else np.nan

    avg_loss = losses.mean() if len(losses) else np.nan

    payoff = avg_win / abs(avg_loss) if len(losses) else np.nan

    # --------------------------------------------------------
    # Daily equity
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
    # Trade t-stat
    # --------------------------------------------------------

    r_std = r.std(ddof=1)

    t_stat = expectancy / (r_std / math.sqrt(n)) if r_std > 0 else np.nan

    return {
        "trades": n,
        "sessions": trades["session_date"].nunique(),
        "total_R": r.sum(),
        "expectancy_R": expectancy,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown_R": max_dd,
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "trade_t_stat": t_stat,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "payoff_ratio": payoff,
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
        "start_date": trades["session_date"].min(),
        "end_date": trades["session_date"].max(),
    }


# ============================================================
# EXECUTE ONE SESSION
# ============================================================


def execute_session(
    session_df: pd.DataFrame,
    or_row: pd.Series,
) -> dict | None:

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

    t = session_df["timestamp ET"].dt.strftime("%H:%M")

    entry_window = session_df[(t >= OR_END) & (t < ENTRY_CUTOFF)].copy()

    entry_window = entry_window.sort_values("timestamp ET")

    if len(entry_window) == 0:
        return None

    # --------------------------------------------------------
    # First breakout
    # --------------------------------------------------------

    direction = None
    entry_timestamp = None
    entry_price = None
    breakout_index = None

    for idx, (_, bar) in enumerate(entry_window.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        long_break = high >= or_high

        short_break = low <= or_low

        # Both sides touched in one
        # minute -> sequence unknown.
        if long_break and short_break:
            if SKIP_BOTH_BREAKS:
                return None

        if long_break:
            direction = "LONG"

            entry_timestamp = bar["timestamp ET"]

            entry_price = or_high

            breakout_index = idx

            break

        if short_break:
            direction = "SHORT"

            entry_timestamp = bar["timestamp ET"]

            entry_price = or_low

            breakout_index = idx

            break

    if direction is None:
        return None

    # --------------------------------------------------------
    # Risk levels
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
    # Manage from NEXT BAR
    #
    # This matches the baseline used in the previous
    # execution audit.
    # --------------------------------------------------------

    later = session_df[session_df["timestamp ET"] > entry_timestamp].copy()

    later = later.sort_values("timestamp ET")

    exit_time = None
    exit_price = None
    exit_reason = None

    for _, bar in later.iterrows():
        high = float(bar["high"])

        low = float(bar["low"])

        if direction == "LONG":
            hit_stop = low <= stop_price

            hit_target = high >= target_price

        else:
            hit_stop = high >= stop_price

            hit_target = low <= target_price

        # ----------------------------------------------------
        # Both hit
        # ----------------------------------------------------

        if hit_stop and hit_target:
            exit_time = bar["timestamp ET"]

            exit_price = stop_price

            exit_reason = "STOP_AND_TARGET_SAME_BAR_STOP_FIRST"

            break

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        if hit_stop:
            exit_time = bar["timestamp ET"]

            exit_price = stop_price

            exit_reason = "INITIAL_STOP"

            break

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        if hit_target:
            exit_time = bar["timestamp ET"]

            exit_price = target_price

            exit_reason = "TAKE_PROFIT"

            break

    # --------------------------------------------------------
    # RTH close
    # --------------------------------------------------------

    if exit_price is None:
        if not FORCE_EXIT_RTH_CLOSE:
            return None

        if len(session_df) == 0:
            return None

        last_bar = session_df.iloc[-1]

        exit_time = last_bar["timestamp ET"]

        exit_price = float(last_bar["close"])

        exit_reason = "RTH_CLOSE"

    # --------------------------------------------------------
    # P&L
    # --------------------------------------------------------

    if direction == "LONG":
        raw_points = exit_price - entry_price

    else:
        raw_points = entry_price - exit_price

    net_R = raw_points / risk_points

    return {
        "session_date": or_row["session_date"],
        "direction": direction,
        "entry_timestamp": entry_timestamp,
        "exit_timestamp": exit_time,
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_range,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_points": risk_points,
        "raw_points": raw_points,
        "net_R": net_R,
        "exit_reason": exit_reason,
        "holding_minutes": (exit_time - entry_timestamp).total_seconds() / 60.0,
    }


# ============================================================
# BACKTEST
# ============================================================


def run_backtest(
    rth: pd.DataFrame,
    opening_ranges: pd.DataFrame,
) -> pd.DataFrame:

    valid = opening_ranges[opening_ranges["valid_or"]].copy()

    or_map = {row["session_date"]: row for _, row in valid.iterrows()}

    trades = []

    for session_date, session_df in rth.groupby(
        "session_date",
        sort=True,
    ):
        if session_date not in or_map:
            continue

        or_row = or_map[session_date]

        trade = execute_session(
            session_df,
            or_row,
        )

        if trade is not None:
            trades.append(trade)

    if not trades:
        return pd.DataFrame()

    trades = pd.DataFrame(trades)

    trades = trades.sort_values("entry_timestamp").reset_index(drop=True)

    return trades


# ============================================================
# YEARLY
# ============================================================


def yearly_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    if len(trades) == 0:
        return pd.DataFrame()

    tmp = trades.copy()

    tmp["year"] = pd.to_datetime(tmp["session_date"]).dt.year

    rows = []

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

        rows.append(
            {
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

    if len(trades) == 0:
        return pd.DataFrame()

    rows = []

    for side, sub in trades.groupby(
        "direction",
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
# PERIOD REPORT
# ============================================================


def report_period(
    trades: pd.DataFrame,
    label: str,
) -> dict:

    if len(trades) == 0:
        return {
            "period": label,
            "trades": 0,
            "total_R": np.nan,
            "expectancy_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "daily_sharpe": np.nan,
            "trade_t_stat": np.nan,
            "long_trades": 0,
            "short_trades": 0,
        }

    m = calculate_metrics(trades)

    return {
        "period": label,
        "trades": m["trades"],
        "sessions": m["sessions"],
        "total_R": m["total_R"],
        "expectancy_R": m["expectancy_R"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "max_drawdown_R": m["max_drawdown_R"],
        "daily_sharpe": m["daily_sharpe"],
        "daily_sortino": m["daily_sortino"],
        "trade_t_stat": m["trade_t_stat"],
        "long_trades": m["long_trades"],
        "short_trades": m["short_trades"],
        "TP_exits": m["TP_exits"],
        "stop_exits": m["stop_exits"],
        "RTH_close_exits": m["RTH_close_exits"],
        "start_date": m["start_date"],
        "end_date": m["end_date"],
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("#" * 80)
    print("# ORB RECONCILIATION TEST")
    print("#" * 80)

    print()
    print("External-study OOS:")

    print(f"    {EXTERNAL_OOS_START} -> {EXTERNAL_OOS_END}")

    print()
    print("Fixed strategy:")

    print("    OR = 30 minutes")

    print("    RR = 2.0")

    print("    Entry cutoff = 11:00 ET")

    print("    Breakout = TOUCH")

    print("    Max trades/session = 1")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    market = load_market()

    rth = prepare_rth(market)

    opening_ranges = build_opening_ranges(rth)

    print()
    print(f"RTH rows: {len(rth):,}")

    print(f"Sessions: {len(opening_ranges):,}")

    print(
        "Valid OR sessions:",
        int(opening_ranges["valid_or"].sum()),
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    trades = run_backtest(
        rth,
        opening_ranges,
    )

    if len(trades) == 0:
        raise RuntimeError("No trades generated.")

    print()
    print(f"Full trades generated: {len(trades):,}")

    # --------------------------------------------------------
    # Full period
    # --------------------------------------------------------

    full_report = report_period(
        trades,
        "FULL_DATASET",
    )

    # --------------------------------------------------------
    # External OOS
    # --------------------------------------------------------

    oos_mask = (trades["session_date"] >= EXTERNAL_OOS_START) & (
        trades["session_date"] <= EXTERNAL_OOS_END
    )

    oos = trades.loc[oos_mask].copy()

    oos_report = report_period(
        oos,
        "EXTERNAL_OOS_2020-06-23_to_2026-06-19",
    )

    # --------------------------------------------------------
    # Pre-OOS
    # --------------------------------------------------------

    pre_oos = trades[trades["session_date"] < EXTERNAL_OOS_START].copy()

    pre_report = report_period(
        pre_oos,
        "PRE_EXTERNAL_OOS",
    )

    # --------------------------------------------------------
    # Post-OOS
    # --------------------------------------------------------

    post_oos = trades[trades["session_date"] > EXTERNAL_OOS_END].copy()

    post_report = report_period(
        post_oos,
        "POST_EXTERNAL_OOS",
    )

    summary = pd.DataFrame(
        [
            full_report,
            pre_report,
            oos_report,
            post_report,
        ]
    )

    # --------------------------------------------------------
    # Yearly OOS
    # --------------------------------------------------------

    oos_yearly = yearly_metrics(oos)

    # --------------------------------------------------------
    # Side OOS
    # --------------------------------------------------------

    oos_side = side_metrics(oos)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    oos_yearly.to_csv(
        YEARLY_FILE,
        index=False,
    )

    oos_side.to_csv(
        SIDE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PERIOD COMPARISON")
    print("=" * 80)

    cols = [
        "period",
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

    print(summary[cols].to_string(index=False))

    # --------------------------------------------------------
    # External OOS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EXACT EXTERNAL OOS")
    print("=" * 80)

    print(oos_report)

    print()
    print("=" * 80)
    print("OOS YEARLY RESULTS")
    print("=" * 80)

    print(oos_yearly.to_string(index=False))

    print()
    print("=" * 80)
    print("OOS LONG / SHORT")
    print("=" * 80)

    print(oos_side.to_string(index=False))

    # --------------------------------------------------------
    # Comparison to screenshot
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("REFERENCE FROM EXTERNAL STUDY")
    print("=" * 80)

    print("Period:       2020-06-23 -> 2026-06-19")

    print("Trades:       ~1,461")

    print("Profit Factor: ~1.26")

    print("Max DD:       ~27.0%")

    print("Sharpe:       ~1.48")

    print()
    print("OUR EXACT OOS:")

    print(f"Trades:       {oos_report['trades']:,}")

    print(f"Total R:      {oos_report['total_R']:+.4f}R")

    print(f"Expectancy:   {oos_report['expectancy_R']:+.6f}R")

    print(f"Win rate:     {oos_report['win_rate']:.4%}")

    print(f"PF:           {oos_report['profit_factor']:.4f}")

    print(f"Max DD:       {oos_report['max_drawdown_R']:+.4f}R")

    print(f"Sharpe:       {oos_report['daily_sharpe']:.4f}")

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
    print("RECONCILIATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
