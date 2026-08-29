from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 SHORT CONTINUATION — LOCAL ROBUSTNESS
# ============================================================
#
# PURPOSE
# -------
# We already found a specific conditional short edge:
#
#   HMM State 2
#   AND past_return_30 in lower tail
#   AND directional_pressure_30 in lower tail
#   AND close_location_30 in lower tail
#   AND normalized_momentum_30 in lower tail
#
# This experiment DOES NOT search for a new strategy.
#
# It tests the local parameter surface around that exact setup.
#
# Dimensions:
#
#   Entry selectivity:
#       bottom 10%
#       bottom 12.5%
#       bottom 15%
#       bottom 17.5%
#       bottom 20%
#
#   Reward / risk:
#       0.50R
#       0.67R
#       0.75R
#       1.00R
#       1.10R
#       1.20R
#       1.30R
#       1.50R
#       1.75R
#       2.00R
#
#   Holding period:
#       5
#       10
#       15 bars
#
# IMPORTANT
# ---------
# HMM is contextual only.
# It does NOT mean "State 2 = short".
#
# A trade requires ALL four directional conditions.
#
# Thresholds are calculated using TRAIN only.
# OOS data is never used to calculate thresholds.
#
# Exactly one position may exist at a time.
#
# ============================================================


# ============================================================
# GLOBAL CONFIG
# ============================================================

RANDOM_STATE = 42
N_STATES = 3

STATE = 2
DIRECTION = "short"

STOP_POINTS = 20.0

# Lower-tail selectivity.
#
# 0.90 = bottom 10%
# 0.875 = bottom 12.5%
# 0.85 = bottom 15%
# 0.825 = bottom 17.5%
# 0.80 = bottom 20%
#
QUANTILES = [
    0.90,
    0.875,
    0.85,
    0.825,
    0.80,
]


# Reward / risk values.
#
# Target = STOP_POINTS * RR
#
RR_VALUES = [
    0.50,
    0.67,
    0.75,
    1.00,
    1.10,
    1.20,
    1.30,
    1.50,
    1.75,
    2.00,
]


HORIZONS = [
    5,
    10,
    15,
]


# ============================================================
# CORE FEATURES
# ============================================================

CONDITION_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_rth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    required_columns = [
        "timestamp ET",
        "market_period",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise KeyError("Missing required columns:\n" + "\n".join(missing))

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.isna().all():
        raise ValueError("Could not parse 'timestamp ET'.")

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")

    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = timestamps

    # --------------------------------------------------------
    # RTH only
    # --------------------------------------------------------

    df = df.loc[df["market_period"] == "RTH"].copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values("_timestamp_et")

    df = df.set_index("_timestamp_et")

    df.index.name = "timestamp_et"

    # --------------------------------------------------------
    # Session identifier
    # --------------------------------------------------------

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
):

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


def fit_hmm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
):

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # HMM is fitted ONLY on training data.
    # --------------------------------------------------------

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
# TRAIN-ONLY THRESHOLDS
# ============================================================


def calculate_thresholds(
    train: pd.DataFrame,
    quantile: float,
):

    thresholds = {}

    lower_quantile = 1.0 - quantile

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Thresholds are calculated only from:
    #
    #     TRAIN observations
    #     AND State 2
    #
    # This prevents the threshold from being dominated by
    # other regimes.
    # --------------------------------------------------------

    state_train = train.loc[train["hmm_state"] == STATE]

    for feature in CONDITION_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if values.empty:
            raise ValueError(f"No valid training data for {feature} in State {STATE}.")

        thresholds[feature] = float(values.quantile(lower_quantile))

    return thresholds


# ============================================================
# SIGNAL
# ============================================================


def is_signal(
    row: pd.Series,
    thresholds: dict[str, float],
):

    # --------------------------------------------------------
    # HMM = context.
    # --------------------------------------------------------

    if row["hmm_state"] != STATE:
        return False

    # --------------------------------------------------------
    # ALL four directional conditions must agree.
    # --------------------------------------------------------

    for feature in CONDITION_FEATURES:
        value = row[feature]

        if pd.isna(value):
            return False

        if value > thresholds[feature]:
            return False

    return True


# ============================================================
# BARRIER RESOLUTION
# ============================================================


def resolve_trade(
    session: pd.DataFrame,
    entry_position: int,
    target_points: float,
    horizon: int,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = close[entry_position]

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    target_price = entry_price - target_points

    stop_price = entry_price + STOP_POINTS

    last_position = min(
        entry_position + horizon,
        len(session) - 1,
    )

    # --------------------------------------------------------
    # Walk forward bar-by-bar.
    # --------------------------------------------------------

    for i in range(
        entry_position + 1,
        last_position + 1,
    ):
        target_hit = low[i] <= target_price

        stop_hit = high[i] >= stop_price

        # ----------------------------------------------------
        # If both barriers are inside the same OHLC bar,
        # we cannot know the intrabar sequence.
        #
        # Conservative assumption:
        # treat it as a STOP.
        # ----------------------------------------------------

        if target_hit and stop_hit:
            return {
                "exit_position": i,
                "pnl_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
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

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    return {
        "exit_position": last_position,
        "pnl_points": pnl_points,
        "reason": "timeout",
    }


# ============================================================
# EXECUTION ENGINE
# ============================================================


def run_strategy(
    validation: pd.DataFrame,
    thresholds: dict[str, float],
    rr: float,
    horizon: int,
    quantile: float,
    window_number: int,
):

    target_points = STOP_POINTS * rr

    trades = []

    # --------------------------------------------------------
    # Process each RTH session independently.
    # --------------------------------------------------------

    for session_id, session in validation.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= horizon:
            continue

        positions = session.index

        i = 0

        # ----------------------------------------------------
        # Non-overlapping execution.
        # ----------------------------------------------------

        while i < len(session) - horizon:
            row = session.iloc[i]

            # ------------------------------------------------
            # FLAT unless the complete hypothesis is active.
            # ------------------------------------------------

            if not is_signal(
                row,
                thresholds,
            ):
                i += 1

                continue

            # ------------------------------------------------
            # ENTER SHORT.
            # ------------------------------------------------

            result = resolve_trade(
                session=session,
                entry_position=i,
                target_points=target_points,
                horizon=horizon,
            )

            pnl_R = result["pnl_points"] / STOP_POINTS

            trades.append(
                {
                    "window": (window_number),
                    "quantile": (quantile),
                    "rr": rr,
                    "target_points": (target_points),
                    "stop_points": (STOP_POINTS),
                    "horizon": (horizon),
                    "entry_timestamp": (positions[i]),
                    "exit_timestamp": (positions[result["exit_position"]]),
                    "pnl_points": (result["pnl_points"]),
                    "pnl_R": pnl_R,
                    "exit_reason": (result["reason"]),
                    "holding_bars": (result["exit_position"] - i),
                }
            )

            # ------------------------------------------------
            # CRITICAL:
            #
            # No new signal can be entered until the current
            # trade has finished.
            # ------------------------------------------------

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

    pnl = trades["pnl_R"].astype(float)

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    else:
        profit_factor = np.inf

    # --------------------------------------------------------
    # Equity / drawdown
    # --------------------------------------------------------

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Losing streak
    # --------------------------------------------------------

    longest_losing_streak = 0
    current_streak = 0

    for value in pnl:
        if value < 0:
            current_streak += 1

            longest_losing_streak = max(
                longest_losing_streak,
                current_streak,
            )

        else:
            current_streak = 0

    # --------------------------------------------------------
    # Daily statistics
    # --------------------------------------------------------

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    if len(daily) > 1 and daily.std(ddof=1) > 0:
        daily_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)

    else:
        daily_sharpe = np.nan

    # --------------------------------------------------------
    # Daily profitability
    # --------------------------------------------------------

    profitable_days = int((daily > 0).sum())

    losing_days = int((daily < 0).sum())

    return {
        "trades": len(trades),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": (pnl > 0).mean(),
        "mean_R": (pnl.mean()),
        "median_R": (pnl.median()),
        "total_R": (pnl.sum()),
        "profit_factor": (profit_factor),
        "max_drawdown_R": (max_drawdown),
        "longest_losing_streak": (longest_losing_streak),
        "average_holding_bars": (trades["holding_bars"].mean()),
        "profitable_days": (profitable_days),
        "losing_days": (losing_days),
        "daily_sharpe": (daily_sharpe),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 SHORT CONTINUATION — LOCAL ROBUSTNESS")

    print("=" * 110)

    print("\nFIXED STRUCTURAL HYPOTHESIS:")

    print("HMM State 2")

    print("AND past_return_30 in lower tail")

    print("AND directional_pressure_30 in lower tail")

    print("AND close_location_30 in lower tail")

    print("AND normalized_momentum_30 in lower tail")

    print("\nTESTING:")

    print("Lower-tail thresholds: 10%, 12.5%, 15%, 17.5%, 20%")

    print("RR: 0.50 → 2.00")

    print("Horizons: 5, 10, 15 bars")

    print("\nNo HMM directional assumption.")

    print("HMM is contextual only.")

    print("All four conditions are required.")

    print("One position at a time.")

    print("Thresholds learned from TRAIN only.")

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    df = prepare_rth(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    # ========================================================
    # FEATURES
    # ========================================================

    print("\n=== ADDING DIRECTIONAL FEATURES ===")

    df = add_directional_features(df)

    # ========================================================
    # WALK-FORWARD WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_metrics = []
    all_trades = []

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    for (
        window_number,
        (
            train_start,
            validation_start,
            validation_end,
        ),
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

        print(f"Train observations: {len(train)}")

        print(f"Validation observations: {len(validation)}")

        # ====================================================
        # HMM
        # ====================================================

        (
            train,
            validation,
        ) = fit_hmm(
            train,
            validation,
        )

        print("\nValidation regime proportions:")

        print(
            validation["hmm_state"]
            .value_counts(normalize=True)
            .sort_index()
            .to_string()
        )

        # ====================================================
        # PARAMETER GRID
        # ====================================================

        for quantile in QUANTILES:
            thresholds = calculate_thresholds(
                train,
                quantile,
            )

            print(f"\nLower-tail selectivity: {(1 - quantile) * 100:.1f}%")

            print(f"past_return_30 <= {thresholds['past_return_30']:.8f}")

            print(
                f"directional_pressure_30 <= "
                f"{thresholds['directional_pressure_30']:.8f}"
            )

            print(f"close_location_30 <= {thresholds['close_location_30']:.8f}")

            print(
                f"normalized_momentum_30 <= {thresholds['normalized_momentum_30']:.8f}"
            )

            # ------------------------------------------------
            # RR
            # ------------------------------------------------

            for rr in RR_VALUES:
                # --------------------------------------------
                # Horizon
                # --------------------------------------------

                for horizon in HORIZONS:
                    trades = run_strategy(
                        validation=validation,
                        thresholds=thresholds,
                        rr=rr,
                        horizon=horizon,
                        quantile=quantile,
                        window_number=window_number,
                    )

                    metrics = calculate_metrics(trades)

                    if metrics is None:
                        continue

                    metrics.update(
                        {
                            "window": (window_number),
                            "quantile": (quantile),
                            "tail_percent": ((1 - quantile) * 100),
                            "rr": rr,
                            "target_points": (STOP_POINTS * rr),
                            "horizon": (horizon),
                        }
                    )

                    all_metrics.append(metrics)

                    if not trades.empty:
                        all_trades.append(trades)

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results = pd.DataFrame(all_metrics)

    if results.empty:
        print("\nNO VALID RESULTS.")

        return

    trades = (
        pd.concat(
            all_trades,
            ignore_index=True,
        )
        if all_trades
        else pd.DataFrame()
    )

    # ========================================================
    # FULL PARAMETER SURFACE
    # ========================================================

    print("\n" + "=" * 110)

    print("LOCAL PARAMETER ROBUSTNESS")

    print("=" * 110)

    summary = (
        results.groupby(
            [
                "quantile",
                "rr",
                "horizon",
            ]
        )
        .agg(
            windows=(
                "window",
                "count",
            ),
            total_trades=(
                "trades",
                "sum",
            ),
            median_WR=(
                "win_rate",
                "median",
            ),
            mean_WR=(
                "win_rate",
                "mean",
            ),
            median_R=(
                "mean_R",
                "median",
            ),
            mean_R=(
                "mean_R",
                "mean",
            ),
            median_PF=(
                "profit_factor",
                "median",
            ),
            mean_PF=(
                "profit_factor",
                "mean",
            ),
            total_R=(
                "total_R",
                "sum",
            ),
            worst_DD=(
                "max_drawdown_R",
                "min",
            ),
            worst_streak=(
                "longest_losing_streak",
                "max",
            ),
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

    # ========================================================
    # POSITIVE IN ALL WINDOWS
    # ========================================================

    print("\n" + "=" * 110)

    print("CONFIGURATIONS POSITIVE IN ALL OOS WINDOWS")

    print("=" * 110)

    consistency = (
        results.groupby(
            [
                "quantile",
                "rr",
                "horizon",
            ]
        )
        .agg(
            windows=(
                "window",
                "count",
            ),
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
            worst_streak=(
                "longest_losing_streak",
                "max",
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
        print("NO CONFIGURATION WAS POSITIVE IN ALL FOUR OOS WINDOWS.")

    else:
        print(robust.to_string(index=False))

    # ========================================================
    # RR SURFACE
    # ========================================================

    print("\n" + "=" * 110)

    print("RR SURFACE — MEDIAN OOS EXPECTANCY")

    print("=" * 110)

    rr_surface = summary.pivot_table(
        index=["quantile"],
        columns=[
            "rr",
            "horizon",
        ],
        values="median_R",
    )

    print(rr_surface.to_string())

    # ========================================================
    # QUANTILE SURFACE
    # ========================================================

    print("\n" + "=" * 110)

    print("QUANTILE × RR — MEDIAN OOS EXPECTANCY")

    print("=" * 110)

    for horizon in HORIZONS:
        print(f"\nHORIZON = {horizon} BARS")

        surface = summary.loc[summary["horizon"] == horizon].pivot_table(
            index="quantile",
            columns="rr",
            values="median_R",
        )

        print(surface.to_string())

    # ========================================================
    # BEST STABLE REGION
    # ========================================================

    print("\n" + "=" * 110)

    print("STRONGEST STABLE CONFIGURATIONS")

    print("=" * 110)

    stable = (
        consistency.loc[
            (consistency["all_positive"]) & (consistency["total_trades"] >= 500)
        ]
        .sort_values(
            [
                "median_R",
                "median_PF",
            ],
            ascending=False,
        )
        .head(20)
    )

    if stable.empty:
        print("No configuration met the stability criteria.")

    else:
        print(stable.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        "s2_short_local_robustness_results.csv",
        index=False,
    )

    summary.to_csv(
        "s2_short_local_robustness_summary.csv",
        index=False,
    )

    consistency.to_csv(
        "s2_short_local_robustness_consistency.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            "s2_short_local_robustness_trades.csv",
            index=False,
        )

    print("\nSaved:")

    print("s2_short_local_robustness_results.csv")

    print("s2_short_local_robustness_summary.csv")

    print("s2_short_local_robustness_consistency.csv")

    print("s2_short_local_robustness_trades.csv")

    print("\n" + "=" * 110)

    print("LOCAL ROBUSTNESS TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
