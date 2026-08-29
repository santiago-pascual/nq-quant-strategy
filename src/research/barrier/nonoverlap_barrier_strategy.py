from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel

# ============================================================
# CONFIGURATION
# ============================================================

N_STATES = 3
RANDOM_STATE = 42

# Candidate we are testing first.
HMM_STATE = 1
DIRECTION = "short"

HORIZON = 5

STOP_POINTS = 20.0

# 0.75R target.
REWARD_RISK = 0.75

TARGET_POINTS = STOP_POINTS * REWARD_RISK

# Walk-forward structure.
TRAIN_YEARS = 2
VALIDATION_MONTHS = 3


# ============================================================
# DATA PREPARATION
# ============================================================


def get_timestamp_series(
    df: pd.DataFrame,
) -> pd.Series:

    if "timestamp ET" not in df.columns:
        raise KeyError("Expected 'timestamp ET' column.")

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    return timestamps


def prepare_rth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_timestamp_et"] = get_timestamp_series(df)

    if "market_period" not in df.columns:
        raise KeyError("Missing 'market_period' column.")

    df = df.loc[df["market_period"] == "RTH"].copy()

    df = df.sort_values("_timestamp_et")

    df = df.set_index("_timestamp_et")

    df.index.name = "timestamp_et"

    if "session_date" in df.columns:
        df["_session_id"] = df["session_date"].astype(str)

    else:
        df["_session_id"] = df.index.date

    return df


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    df: pd.DataFrame,
) -> list[
    tuple[
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:

    start = df.index.min()
    end = df.index.max()

    validation_start = start + pd.DateOffset(years=TRAIN_YEARS)

    windows = []

    while validation_start < end:
        train_start = validation_start - pd.DateOffset(years=TRAIN_YEARS)

        validation_end = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)

        validation_end = min(
            validation_end,
            end,
        )

        windows.append(
            (
                train_start,
                validation_start,
                validation_end,
            )
        )

        validation_start = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)

    return windows


# ============================================================
# HMM
# ============================================================


def fit_hmm_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    return (
        train,
        validation,
    )


# ============================================================
# BARRIER RESOLUTION
# ============================================================


def resolve_short_trade(
    session: pd.DataFrame,
    entry_position: int,
) -> dict | None:
    """
    Enter SHORT at the close of entry_position.

    Target:
        entry - TARGET_POINTS

    Stop:
        entry + STOP_POINTS

    Maximum holding period:
        HORIZON bars.

    Returns exactly one trade.

    Important:
    If TP and SL are both touched inside the same
    1-minute candle, the ordering is unknown from OHLC.
    We conservatively classify that trade as a loss.
    """

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = close[entry_position]

    target_price = entry_price - TARGET_POINTS

    stop_price = entry_price + STOP_POINTS

    last_position = min(
        entry_position + HORIZON,
        len(session) - 1,
    )

    for position in range(
        entry_position + 1,
        last_position + 1,
    ):
        bar_high = high[position]
        bar_low = low[position]

        target_hit = bar_low <= target_price

        stop_hit = bar_high >= stop_price

        # Conservative assumption:
        # if both barriers occur in the same
        # minute, classify as STOP.
        if target_hit and stop_hit:
            return {
                "exit_position": position,
                "exit_reason": "both_hit_conservative_stop",
                "outcome": -1,
                "pnl_points": -STOP_POINTS,
            }

        if target_hit:
            return {
                "exit_position": position,
                "exit_reason": "target",
                "outcome": 1,
                "pnl_points": TARGET_POINTS,
            }

        if stop_hit:
            return {
                "exit_position": position,
                "exit_reason": "stop",
                "outcome": -1,
                "pnl_points": -STOP_POINTS,
            }

    # --------------------------------------------------------
    # Neither barrier was hit.
    #
    # Exit at the close of the final holding bar.
    # --------------------------------------------------------

    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    if pnl_points > 0:
        outcome = 1

    elif pnl_points < 0:
        outcome = -1

    else:
        outcome = 0

    return {
        "exit_position": last_position,
        "exit_reason": "timeout",
        "outcome": outcome,
        "pnl_points": pnl_points,
    }


# ============================================================
# NON-OVERLAPPING STRATEGY
# ============================================================


def run_strategy(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    trades = []

    # --------------------------------------------------------
    # Only State 1 is allowed.
    # --------------------------------------------------------

    validation = validation.loc[validation["hmm_state"] == HMM_STATE].copy()

    if validation.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Process each RTH session separately.
    #
    # This prevents a position from crossing
    # the RTH session boundary.
    # --------------------------------------------------------

    for session_id, session in validation.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= HORIZON:
            continue

        positions = session.index

        i = 0

        # ----------------------------------------------------
        # THE CRITICAL RULE:
        #
        # After entering a trade, jump directly to the
        # bar AFTER the exit.
        #
        # Therefore trades CANNOT overlap.
        # ----------------------------------------------------

        while i < len(session) - HORIZON:
            entry_timestamp = positions[i]

            entry_price = float(session.iloc[i]["close"])

            result = resolve_short_trade(
                session=session,
                entry_position=i,
            )

            if result is None:
                break

            exit_position = result["exit_position"]

            exit_timestamp = positions[exit_position]

            trades.append(
                {
                    "entry_timestamp": (entry_timestamp),
                    "exit_timestamp": (exit_timestamp),
                    "session_id": (session_id),
                    "direction": (DIRECTION),
                    "hmm_state": (HMM_STATE),
                    "entry_price": (entry_price),
                    "target_points": (TARGET_POINTS),
                    "stop_points": (STOP_POINTS),
                    "horizon": (HORIZON),
                    "reward_risk": (REWARD_RISK),
                    "outcome": (result["outcome"]),
                    "pnl_points": (result["pnl_points"]),
                    "exit_reason": (result["exit_reason"]),
                    "holding_bars": (exit_position - i),
                }
            )

            # ------------------------------------------------
            # Jump past the entire trade.
            # ------------------------------------------------

            i = exit_position + 1

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict:

    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "total_R": 0.0,
            "mean_R": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
            "longest_losing_streak": 0,
            "average_holding_bars": np.nan,
        }

    pnl = trades["pnl_points"].astype(float)

    pnl_R = pnl / STOP_POINTS

    wins = int((pnl > 0).sum())

    losses = int((pnl < 0).sum())

    resolved = wins + losses

    win_rate = wins / resolved if resolved > 0 else np.nan

    gross_profit = pnl_R[pnl_R > 0].sum()

    gross_loss = -pnl_R[pnl_R < 0].sum()

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = pnl_R.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = drawdown.min()

    longest_losing_streak = 0
    current_streak = 0

    for value in pnl_R:
        if value < 0:
            current_streak += 1

            longest_losing_streak = max(
                longest_losing_streak,
                current_streak,
            )

        else:
            current_streak = 0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_R": pnl_R.sum(),
        "mean_R": pnl_R.mean(),
        "profit_factor": profit_factor,
        "max_drawdown_R": max_drawdown,
        "longest_losing_streak": (longest_losing_streak),
        "average_holding_bars": (trades["holding_bars"].mean()),
    }


# ============================================================
# DAILY METRICS
# ============================================================


def calculate_daily_metrics(
    trades: pd.DataFrame,
) -> dict:

    if trades.empty:
        return {
            "profitable_days": 0,
            "losing_days": 0,
            "daily_sharpe": np.nan,
            "daily_sortino": np.nan,
            "worst_day_R": np.nan,
        }

    daily = (
        trades.set_index("entry_timestamp")["pnl_points"].resample("1D").sum()
        / STOP_POINTS
    )

    profitable_days = int((daily > 0).sum())

    losing_days = int((daily < 0).sum())

    daily_std = daily.std(ddof=1)

    if daily_std > 0 and len(daily) > 1:
        daily_sharpe = daily.mean() / daily_std * np.sqrt(252)

    else:
        daily_sharpe = np.nan

    downside = daily[daily < 0]

    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan

    if np.isfinite(downside_std) and downside_std > 0:
        daily_sortino = daily.mean() / downside_std * np.sqrt(252)

    else:
        daily_sortino = np.nan

    return {
        "profitable_days": (profitable_days),
        "losing_days": (losing_days),
        "daily_sharpe": (daily_sharpe),
        "daily_sortino": (daily_sortino),
        "worst_day_R": (daily.min()),
    }


# ============================================================
# WINDOW
# ============================================================


def evaluate_window(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    window_number: int,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    print(f"\n{'#' * 100}")

    print(f"WINDOW {window_number}")

    print(f"{'#' * 100}")

    print(f"Train observations: {len(train)}")

    print(f"Validation observations: {len(validation)}")

    # --------------------------------------------------------
    # Fit HMM ONLY on training data.
    # --------------------------------------------------------

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    validation = validation.copy()

    validation["hmm_state"] = model.predict_states(validation)

    print("\nValidation regime proportions:")

    print(validation["hmm_state"].value_counts(normalize=True).sort_index())

    # --------------------------------------------------------
    # Execute the strategy.
    # --------------------------------------------------------

    trades = run_strategy(validation)

    if trades.empty:
        print("\nNo trades.")

        return (
            trades,
            {},
        )

    metrics = calculate_metrics(trades)

    daily_metrics = calculate_daily_metrics(trades)

    metrics.update(daily_metrics)

    print("\nWINDOW RESULTS")

    for key, value in metrics.items():
        print(f"{key:30s}: {value}")

    print("\nExit reasons:")

    print(trades["exit_reason"].value_counts().to_string())

    return (
        trades,
        metrics,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)

    print("NON-OVERLAPPING BARRIER STRATEGY")

    print("STATE 1 / SHORT / 5 BARS / 0.75R / 20 POINT STOP")

    print("=" * 100)

    print("\nThis is a realistic execution diagnostic.")

    print("Exactly ONE position can exist at a time.")

    print("No overlapping trades.")

    print("No parameter switching during a trade.")

    print("HMM is fitted only on the training period.")

    print("No transaction costs or slippage yet.")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"RTH sessions: {rth['_session_id'].nunique()}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    windows = generate_windows(rth)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_trades = []
    all_metrics = []

    # --------------------------------------------------------
    # Walk-forward
    # --------------------------------------------------------

    for window_number, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        train = rth.loc[
            (rth.index >= train_start) & (rth.index < validation_start)
        ].copy()

        validation = rth.loc[
            (rth.index >= validation_start) & (rth.index < validation_end)
        ].copy()

        print(f"\nTrain: {train_start.date()} → {validation_start.date()}")

        print(f"Validation: {validation_start.date()} → {validation_end.date()}")

        trades, metrics = evaluate_window(
            train=train,
            validation=validation,
            window_number=window_number,
        )

        if not trades.empty:
            trades = trades.copy()

            trades["window"] = window_number

            all_trades.append(trades)

        if metrics:
            metrics["window"] = window_number

            all_metrics.append(metrics)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if not all_trades:
        print("\nNo trades generated.")

        return

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    trades = trades.sort_values("entry_timestamp")

    trades = trades.reset_index(drop=True)

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_metrics = calculate_metrics(trades)

    overall_daily = calculate_daily_metrics(trades)

    overall_metrics.update(overall_daily)

    print("\n" + "=" * 100)

    print("COMBINED OOS RESULTS")

    print("=" * 100)

    for key, value in overall_metrics.items():
        print(f"{key:30s}: {value}")

    # --------------------------------------------------------
    # Window comparison
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("WALK-FORWARD WINDOW COMPARISON")

    print("=" * 100)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        columns = [
            "window",
            "trades",
            "win_rate",
            "total_R",
            "mean_R",
            "profit_factor",
            "max_drawdown_R",
            "longest_losing_streak",
            "profitable_days",
            "losing_days",
            "daily_sharpe",
            "daily_sortino",
            "worst_day_R",
        ]

        print(metrics_df[columns].to_string(index=False))

    # --------------------------------------------------------
    # Exit analysis
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("EXIT ANALYSIS")

    print("=" * 100)

    print(trades["exit_reason"].value_counts().to_string())

    # --------------------------------------------------------
    # Holding period
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("HOLDING PERIOD")

    print("=" * 100)

    print(trades["holding_bars"].describe().to_string())

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = "nonoverlap_barrier_strategy_results.csv"

    trades.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved trades to:")

    print(output_path)

    print("\n" + "=" * 100)

    print("NON-OVERLAPPING STRATEGY TEST COMPLETE")

    print("=" * 100)

    print("\nIMPORTANT:")

    print("This is still a research test.")

    print("No slippage.")

    print("No commissions.")

    print("No funded-account rules.")

    print("No position sizing.")

    print("No live execution.")


if __name__ == "__main__":
    main()
