from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
N_STATES = 3

STATE = 2
DIRECTION = "short"

# The discovered candidate is based on 30-bar conditions.
LOOKBACK = 30

STOP_POINTS = 20.0

# Robustness neighborhood.
QUANTILES = [
    0.70,
    0.75,
    0.80,
    0.85,
]

BARRIERS = [
    ("0.50R", 10.0),
    ("0.67R", 13.4),
    ("0.75R", 15.0),
    ("1.00R", 20.0),
]

HORIZONS = [
    5,
    10,
    15,
]


# ============================================================
# DATA
# ============================================================


def prepare_rth(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    ts = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = ts

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
# WALK FORWARD
# ============================================================


def generate_windows(df):

    start = df.index.min()
    end = df.index.max()

    validation_start = start + pd.DateOffset(years=2)

    windows = []

    while validation_start < end:
        validation_end = min(
            validation_start + pd.DateOffset(months=3),
            end,
        )

        train_start = validation_start - pd.DateOffset(years=2)

        windows.append(
            (
                train_start,
                validation_start,
                validation_end,
            )
        )

        validation_start += pd.DateOffset(months=3)

    return windows


# ============================================================
# HMM
# ============================================================


def fit_hmm(train, validation):

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    return train, validation


# ============================================================
# CONDITION FEATURES
# ============================================================

CONDITION_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


def calculate_thresholds(
    train: pd.DataFrame,
    quantile: float,
):

    thresholds = {}

    lower_q = 1.0 - quantile

    for feature in CONDITION_FEATURES:
        values = (
            train.loc[
                train["hmm_state"] == STATE,
                feature,
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        thresholds[feature] = (float(values.quantile(lower_q)),)

    return {feature: thresholds[feature][0] for feature in CONDITION_FEATURES}


# ============================================================
# SIGNAL
# ============================================================


def is_signal(
    row,
    thresholds,
):

    if row["hmm_state"] != STATE:
        return False

    for feature in CONDITION_FEATURES:
        value = row[feature]

        if pd.isna(value):
            return False

        # SHORT requires all four conditions
        # to be in the lower tail.
        if value > thresholds[feature]:
            return False

    return True


# ============================================================
# BARRIER
# ============================================================


def resolve_trade(
    session,
    entry_position,
    target_points,
    horizon,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry = close[entry_position]

    target = entry - target_points

    stop = entry + STOP_POINTS

    last = min(
        entry_position + horizon,
        len(session) - 1,
    )

    for i in range(
        entry_position + 1,
        last + 1,
    ):
        target_hit = low[i] <= target

        stop_hit = high[i] >= stop

        # Conservative treatment of ambiguous
        # one-minute OHLC bars.
        if target_hit and stop_hit:
            return {
                "exit_position": i,
                "pnl_points": -STOP_POINTS,
                "reason": "both_hit",
            }

        if target_hit:
            return {
                "exit_position": i,
                "pnl_points": target_points,
                "reason": "target",
            }

        if stop_hit:
            return {
                "exit_position": i,
                "pnl_points": -STOP_POINTS,
                "reason": "stop",
            }

    # Timeout.
    exit_price = close[last]

    pnl = entry - exit_price

    return {
        "exit_position": last,
        "pnl_points": pnl,
        "reason": "timeout",
    }


# ============================================================
# EXECUTION
# ============================================================


def run_strategy(
    validation,
    thresholds,
    target_points,
    barrier_name,
    horizon,
    quantile,
    window,
):

    trades = []

    for session_id, session in validation.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= horizon:
            continue

        positions = session.index

        i = 0

        while i < len(session) - horizon:
            row = session.iloc[i]

            # ------------------------------------------------
            # FLAT unless the COMPLETE hypothesis is true.
            # ------------------------------------------------

            if not is_signal(
                row,
                thresholds,
            ):
                i += 1
                continue

            result = resolve_trade(
                session=session,
                entry_position=i,
                target_points=target_points,
                horizon=horizon,
            )

            pnl_R = result["pnl_points"] / STOP_POINTS

            trades.append(
                {
                    "window": window,
                    "quantile": quantile,
                    "barrier": barrier_name,
                    "horizon": horizon,
                    "entry_timestamp": positions[i],
                    "exit_timestamp": positions[result["exit_position"]],
                    "pnl_points": result["pnl_points"],
                    "pnl_R": pnl_R,
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - i),
                }
            )

            # No overlapping trades.
            i = result["exit_position"] + 1

    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades: pd.DataFrame,
):

    if trades.empty:
        return None

    pnl = trades["pnl_R"]

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = -losses.sum()

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    streak = 0
    current = 0

    for value in pnl:
        if value < 0:
            current += 1

            streak = max(
                streak,
                current,
            )

        else:
            current = 0

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    if len(daily) > 1 and daily.std(ddof=1) > 0:
        daily_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)

    else:
        daily_sharpe = np.nan

    return {
        "trades": len(trades),
        "win_rate": ((pnl > 0).mean()),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "profit_factor": pf,
        "max_drawdown_R": drawdown.min(),
        "losing_streak": streak,
        "daily_sharpe": daily_sharpe,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 SHORT CONTINUATION — ROBUSTNESS TEST")
    print("=" * 110)

    print("\nFixed hypothesis:")

    print("HMM State 2")

    print("AND past_return_30 in lower tail")

    print("AND directional_pressure_30 in lower tail")

    print("AND close_location_30 in lower tail")

    print("AND normalized_momentum_30 in lower tail")

    print("\nOnly nearby parameters are being tested.")

    print("Thresholds are learned from TRAIN only.")

    print("Trades are non-overlapping.")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = load_data()

    df = prepare_rth(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    print("\n=== ADDING DIRECTIONAL FEATURES ===")

    df = add_directional_features(df)

    windows = generate_windows(df)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_trades = []
    all_metrics = []

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    for window_number, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        print("\n" + "#" * 110)

        print(f"WINDOW {window_number}")

        print("#" * 110)

        train = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        print(f"Train: {train_start.date()} → {validation_start.date()}")

        print(f"Validation: {validation_start.date()} → {validation_end.date()}")

        # ----------------------------------------------------
        # HMM fitted ONLY on train.
        # ----------------------------------------------------

        train, validation = fit_hmm(
            train,
            validation,
        )

        # ----------------------------------------------------
        # PARAMETER NEIGHBORHOOD
        # ----------------------------------------------------

        for quantile in QUANTILES:
            thresholds = calculate_thresholds(
                train,
                quantile,
            )

            print(f"\nQuantile: {quantile:.2f}")

            for feature in CONDITION_FEATURES:
                print(f"  {feature:30s} <= {thresholds[feature]:.8f}")

            for barrier_name, target_points in BARRIERS:
                for horizon in HORIZONS:
                    trades = run_strategy(
                        validation=validation,
                        thresholds=thresholds,
                        target_points=target_points,
                        barrier_name=barrier_name,
                        horizon=horizon,
                        quantile=quantile,
                        window=window_number,
                    )

                    result = calculate_metrics(trades)

                    if result is None:
                        continue

                    result.update(
                        {
                            "window": window_number,
                            "quantile": quantile,
                            "barrier": barrier_name,
                            "horizon": horizon,
                        }
                    )

                    all_metrics.append(result)

                    if not trades.empty:
                        all_trades.append(trades)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(all_metrics)

    trades = (
        pd.concat(
            all_trades,
            ignore_index=True,
        )
        if all_trades
        else pd.DataFrame()
    )

    print("\n" + "=" * 110)

    print("PARAMETER ROBUSTNESS RESULTS")

    print("=" * 110)

    summary = (
        results.groupby(
            [
                "quantile",
                "barrier",
                "horizon",
            ]
        )
        .agg(
            windows=("window", "count"),
            total_trades=("trades", "sum"),
            median_WR=("win_rate", "median"),
            mean_WR=("win_rate", "mean"),
            median_R=("mean_R", "median"),
            mean_R=("mean_R", "mean"),
            median_PF=("profit_factor", "median"),
            mean_PF=("profit_factor", "mean"),
            total_R=("total_R", "sum"),
            worst_DD=("max_drawdown_R", "min"),
            worst_streak=("losing_streak", "max"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "median_R",
            "median_PF",
        ],
        ascending=False,
    )

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # Positive across ALL windows
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("CANDIDATES POSITIVE IN EVERY OOS WINDOW")

    print("=" * 110)

    consistency = (
        results.groupby(
            [
                "quantile",
                "barrier",
                "horizon",
            ]
        )
        .agg(
            windows=("window", "count"),
            positive_windows=(
                "mean_R",
                lambda x: int((x > 0).sum()),
            ),
            median_R=(
                "mean_R",
                "median",
            ),
            median_PF=(
                "profit_factor",
                "median",
            ),
            total_trades=(
                "trades",
                "sum",
            ),
            worst_DD=(
                "max_drawdown_R",
                "min",
            ),
        )
        .reset_index()
    )

    consistency["all_positive"] = (
        consistency["positive_windows"] == consistency["windows"]
    )

    robust = consistency.loc[consistency["all_positive"]].sort_values(
        [
            "median_R",
            "median_PF",
        ],
        ascending=False,
    )

    if robust.empty:
        print("NO CONFIGURATION WAS POSITIVE IN ALL WINDOWS.")

    else:
        print(robust.to_string(index=False))

    # --------------------------------------------------------
    # Parameter surface
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("MEDIAN OOS EXPECTANCY SURFACE")

    print("=" * 110)

    surface = summary.pivot_table(
        index="quantile",
        columns=[
            "barrier",
            "horizon",
        ],
        values="median_R",
    )

    print(surface.to_string())

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        "s2_short_robustness_results.csv",
        index=False,
    )

    summary.to_csv(
        "s2_short_robustness_summary.csv",
        index=False,
    )

    consistency.to_csv(
        "s2_short_robustness_consistency.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            "s2_short_robustness_trades.csv",
            index=False,
        )

    print("\nSaved:")

    print("s2_short_robustness_results.csv")

    print("s2_short_robustness_summary.csv")

    print("s2_short_robustness_consistency.csv")

    print("s2_short_robustness_trades.csv")

    print("\n" + "=" * 110)

    print("ROBUSTNESS TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
