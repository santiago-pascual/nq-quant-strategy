from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features

# ============================================================
# CONFIG
# ============================================================

N_STATES = 3
RANDOM_STATE = 42

HORIZONS = [5, 10, 15]

# We deliberately use a small number of fixed barrier
# structures. No optimization on OOS.
BARRIER_CONFIGS = [
    {"name": "0.50R", "stop": 20.0, "target": 10.0},
    {"name": "0.67R", "stop": 20.0, "target": 13.4},
    {"name": "0.75R", "stop": 20.0, "target": 15.0},
    {"name": "1.00R", "stop": 20.0, "target": 20.0},
]

# Evidence-backed hypotheses.
#
# q = upper/lower quintile calculated ONLY from TRAIN.
HYPOTHESES = [
    {
        "name": "S1_P10_UP",
        "state": 1,
        "feature": "past_return_10",
        "side": "upper",
        "direction": "long",
    },
    {
        "name": "S1_P15_UP",
        "state": 1,
        "feature": "past_return_15",
        "side": "upper",
        "direction": "long",
    },
    {
        "name": "S2_P15_UP",
        "state": 2,
        "feature": "past_return_15",
        "side": "upper",
        "direction": "long",
    },
    {
        "name": "S2_P30_UP",
        "state": 2,
        "feature": "past_return_30",
        "side": "upper",
        "direction": "long",
    },
    {
        "name": "S1_P10_DOWN",
        "state": 1,
        "feature": "past_return_10",
        "side": "lower",
        "direction": "short",
    },
    {
        "name": "S1_P15_DOWN",
        "state": 1,
        "feature": "past_return_15",
        "side": "lower",
        "direction": "short",
    },
    {
        "name": "S2_P15_DOWN",
        "state": 2,
        "feature": "past_return_15",
        "side": "lower",
        "direction": "short",
    },
    {
        "name": "S2_P30_DOWN",
        "state": 2,
        "feature": "past_return_30",
        "side": "lower",
        "direction": "short",
    },
]


# ============================================================
# DATA
# ============================================================


def prepare_rth(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "timestamp ET" not in df.columns:
        raise KeyError("Missing 'timestamp ET'.")

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = timestamps

    if "market_period" not in df.columns:
        raise KeyError("Missing 'market_period'.")

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


def add_hmm_states(
    train,
    validation,
):

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
# TRAIN-ONLY QUANTILES
# ============================================================


def get_threshold(
    train: pd.DataFrame,
    feature: str,
    side: str,
) -> float:

    values = (
        train[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if side == "upper":
        return float(values.quantile(0.80))

    if side == "lower":
        return float(values.quantile(0.20))

    raise ValueError(f"Unknown side: {side}")


# ============================================================
# SIGNAL
# ============================================================


def hypothesis_signal(
    row,
    hypothesis,
    threshold,
):

    if row["hmm_state"] != hypothesis["state"]:
        return False

    value = row[hypothesis["feature"]]

    if pd.isna(value):
        return False

    if hypothesis["side"] == "upper":
        return value >= threshold

    return value <= threshold


# ============================================================
# BARRIER
# ============================================================


def resolve_trade(
    session,
    entry_position,
    direction,
    stop_points,
    target_points,
    horizon,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry = close[entry_position]

    if direction == "long":
        target = entry + target_points
        stop = entry - stop_points

    else:
        target = entry - target_points
        stop = entry + stop_points

    last = min(
        entry_position + horizon,
        len(session) - 1,
    )

    for i in range(
        entry_position + 1,
        last + 1,
    ):
        hit_target = low[i] <= target <= high[i]

        hit_stop = low[i] <= stop <= high[i]

        # Ambiguous OHLC ordering.
        # Conservative treatment = loss.
        if hit_target and hit_stop:
            return {
                "exit_position": i,
                "pnl_points": -stop_points,
                "reason": "both_hit",
            }

        if direction == "long":
            if low[i] <= stop:
                return {
                    "exit_position": i,
                    "pnl_points": -stop_points,
                    "reason": "stop",
                }

            if high[i] >= target:
                return {
                    "exit_position": i,
                    "pnl_points": target_points,
                    "reason": "target",
                }

        else:
            if high[i] >= stop:
                return {
                    "exit_position": i,
                    "pnl_points": -stop_points,
                    "reason": "stop",
                }

            if low[i] <= target:
                return {
                    "exit_position": i,
                    "pnl_points": target_points,
                    "reason": "target",
                }

    exit_price = close[last]

    if direction == "long":
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price

    return {
        "exit_position": last,
        "pnl_points": pnl,
        "reason": "timeout",
    }


# ============================================================
# RUN ONE HYPOTHESIS
# ============================================================


def run_hypothesis(
    validation,
    hypothesis,
    threshold,
    barrier,
    horizon,
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

            if not hypothesis_signal(
                row,
                hypothesis,
                threshold,
            ):
                i += 1
                continue

            result = resolve_trade(
                session=session,
                entry_position=i,
                direction=hypothesis["direction"],
                stop_points=barrier["stop"],
                target_points=barrier["target"],
                horizon=horizon,
            )

            pnl_r = result["pnl_points"] / barrier["stop"]

            trades.append(
                {
                    "entry_timestamp": positions[i],
                    "exit_timestamp": positions[result["exit_position"]],
                    "hypothesis": hypothesis["name"],
                    "direction": hypothesis["direction"],
                    "state": hypothesis["state"],
                    "feature": hypothesis["feature"],
                    "threshold": threshold,
                    "horizon": horizon,
                    "barrier": barrier["name"],
                    "pnl_points": result["pnl_points"],
                    "pnl_R": pnl_r,
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - i),
                }
            )

            # Critical:
            # do not allow another position until
            # this one is completely finished.
            i = result["exit_position"] + 1

    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================


def metrics(trades):

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

    losing_streak = 0
    current = 0

    for x in pnl:
        if x < 0:
            current += 1
            losing_streak = max(
                losing_streak,
                current,
            )
        else:
            current = 0

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    daily_std = daily.std()

    if daily_std > 0:
        daily_sharpe = daily.mean() / daily_std * np.sqrt(252)
    else:
        daily_sharpe = np.nan

    return {
        "trades": len(trades),
        "win_rate": ((pnl > 0).mean()),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "profit_factor": pf,
        "max_drawdown_R": drawdown.min(),
        "losing_streak": losing_streak,
        "daily_sharpe": daily_sharpe,
        "avg_hold": trades["holding_bars"].mean(),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("CONDITIONAL BARRIER HYPOTHESIS TEST")
    print("=" * 110)

    print("\nTesting only previously discovered conditional hypotheses.")

    print("Quantile thresholds are calculated on TRAIN only.")

    print("Trades are non-overlapping.")

    print("No OOS optimization.")

    df = load_data()

    df = prepare_rth(df)

    # --------------------------------------------------------
    # Directional features
    # --------------------------------------------------------

    print("\n=== ADDING DIRECTIONAL FEATURES ===")

    df = add_directional_features(df)

    windows = generate_windows(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"Walk-forward windows: {len(windows)}")

    all_results = []

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
        train = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        print("\n" + "#" * 110)

        print(f"WINDOW {window_number}")

        print("#" * 110)

        print(f"Train: {train_start.date()} → {validation_start.date()}")

        print(f"Validation: {validation_start.date()} → {validation_end.date()}")

        # ----------------------------------------------------
        # Fit HMM only on train.
        # ----------------------------------------------------

        train, validation = add_hmm_states(
            train,
            validation,
        )

        # ----------------------------------------------------
        # Test every predefined hypothesis.
        # ----------------------------------------------------

        for hypothesis in HYPOTHESES:
            threshold = get_threshold(
                train,
                hypothesis["feature"],
                hypothesis["side"],
            )

            for barrier in BARRIER_CONFIGS:
                for horizon in HORIZONS:
                    trades = run_hypothesis(
                        validation=validation,
                        hypothesis=hypothesis,
                        threshold=threshold,
                        barrier=barrier,
                        horizon=horizon,
                    )

                    result = metrics(trades)

                    if result is None:
                        continue

                    result.update(
                        {
                            "window": window_number,
                            "hypothesis": hypothesis["name"],
                            "direction": hypothesis["direction"],
                            "state": hypothesis["state"],
                            "feature": hypothesis["feature"],
                            "threshold": threshold,
                            "barrier": barrier["name"],
                            "horizon": horizon,
                        }
                    )

                    all_results.append(result)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(all_results)

    if results.empty:
        print("\nNo valid hypotheses produced trades.")

        return

    print("\n" + "=" * 110)

    print("OOS HYPOTHESIS RESULTS")

    print("=" * 110)

    summary = (
        results.groupby(
            [
                "hypothesis",
                "direction",
                "barrier",
                "horizon",
            ]
        )
        .agg(
            windows=("window", "count"),
            total_trades=("trades", "sum"),
            mean_WR=("win_rate", "mean"),
            median_mean_R=("mean_R", "median"),
            mean_PF=("profit_factor", "mean"),
            median_PF=("profit_factor", "median"),
            total_R=("total_R", "sum"),
            worst_DD=("max_drawdown_R", "min"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "median_mean_R",
            "median_PF",
        ],
        ascending=False,
    )

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # Strong candidates
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("CANDIDATES WITH POSITIVE MEDIAN OOS EXPECTANCY")

    print("=" * 110)

    positive = summary.loc[summary["median_mean_R"] > 0]

    if positive.empty:
        print("NONE")

    else:
        print(positive.to_string(index=False))

    # --------------------------------------------------------
    # Window consistency
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("WINDOW-BY-WINDOW RESULTS FOR BEST CANDIDATES")

    print("=" * 110)

    for _, row in positive.head(10).iterrows():
        mask = (
            (results["hypothesis"] == row["hypothesis"])
            & (results["barrier"] == row["barrier"])
            & (results["horizon"] == row["horizon"])
        )

        subset = results.loc[
            mask,
            [
                "window",
                "hypothesis",
                "barrier",
                "horizon",
                "trades",
                "win_rate",
                "mean_R",
                "profit_factor",
                "total_R",
                "max_drawdown_R",
            ],
        ]

        print("\n" + subset.to_string(index=False))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        "conditional_barrier_hypothesis_results.csv",
        index=False,
    )

    summary.to_csv(
        "conditional_barrier_hypothesis_summary.csv",
        index=False,
    )

    print("\nSaved:")

    print("conditional_barrier_hypothesis_results.csv")

    print("conditional_barrier_hypothesis_summary.csv")

    print("\n" + "=" * 110)

    print("CONDITIONAL HYPOTHESIS TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
