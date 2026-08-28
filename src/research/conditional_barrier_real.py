from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_STATES = 3

STOP_POINTS = 20.0

BARRIERS = [
    ("0.50R", 10.0),
    ("0.67R", 13.4),
    ("0.75R", 15.0),
    ("1.00R", 20.0),
]

HORIZONS = [5, 10, 15]


# ============================================================
# IMPORTANT
# ============================================================
#
# These are COMPLETE hypotheses.
#
# We are NOT testing:
#
#     HMM state == 1 -> short
#
# Instead, a trade requires:
#
#     REGIME
#     +
#     MOMENTUM CONDITION
#     +
#     DIRECTIONAL PRESSURE
#     +
#     CLOSE LOCATION
#     +
#     NORMALIZED MOMENTUM
#
# to agree.
#
# Quantile thresholds are calculated on TRAIN only.
#
# ============================================================


HYPOTHESES = [
    # --------------------------------------------------------
    # LONG CONTINUATION — STATE 1
    # --------------------------------------------------------
    {
        "name": "S1_LONG_CONTINUATION",
        "state": 1,
        "direction": "long",
        "conditions": [
            ("past_return_10", "upper"),
            ("directional_pressure_10", "upper"),
            ("close_location_10", "upper"),
            ("normalized_momentum_10", "upper"),
        ],
    },
    {
        "name": "S1_LONG_CONTINUATION_15",
        "state": 1,
        "direction": "long",
        "conditions": [
            ("past_return_15", "upper"),
            ("directional_pressure_15", "upper"),
            ("close_location_15", "upper"),
            ("normalized_momentum_15", "upper"),
        ],
    },
    # --------------------------------------------------------
    # LONG CONTINUATION — STATE 2
    # --------------------------------------------------------
    {
        "name": "S2_LONG_CONTINUATION",
        "state": 2,
        "direction": "long",
        "conditions": [
            ("past_return_15", "upper"),
            ("directional_pressure_15", "upper"),
            ("close_location_15", "upper"),
            ("normalized_momentum_15", "upper"),
        ],
    },
    {
        "name": "S2_LONG_CONTINUATION_30",
        "state": 2,
        "direction": "long",
        "conditions": [
            ("past_return_30", "upper"),
            ("directional_pressure_30", "upper"),
            ("close_location_30", "upper"),
            ("normalized_momentum_30", "upper"),
        ],
    },
    # --------------------------------------------------------
    # SHORT CONTINUATION — STATE 1
    # --------------------------------------------------------
    {
        "name": "S1_SHORT_CONTINUATION",
        "state": 1,
        "direction": "short",
        "conditions": [
            ("past_return_10", "lower"),
            ("directional_pressure_10", "lower"),
            ("close_location_10", "lower"),
            ("normalized_momentum_10", "lower"),
        ],
    },
    {
        "name": "S1_SHORT_CONTINUATION_15",
        "state": 1,
        "direction": "short",
        "conditions": [
            ("past_return_15", "lower"),
            ("directional_pressure_15", "lower"),
            ("close_location_15", "lower"),
            ("normalized_momentum_15", "lower"),
        ],
    },
    # --------------------------------------------------------
    # SHORT CONTINUATION — STATE 2
    # --------------------------------------------------------
    {
        "name": "S2_SHORT_CONTINUATION",
        "state": 2,
        "direction": "short",
        "conditions": [
            ("past_return_15", "lower"),
            ("directional_pressure_15", "lower"),
            ("close_location_15", "lower"),
            ("normalized_momentum_15", "lower"),
        ],
    },
    {
        "name": "S2_SHORT_CONTINUATION_30",
        "state": 2,
        "direction": "short",
        "conditions": [
            ("past_return_30", "lower"),
            ("directional_pressure_30", "lower"),
            ("close_location_30", "lower"),
            ("normalized_momentum_30", "lower"),
        ],
    },
]


# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_rth(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "timestamp ET" not in df.columns:
        raise KeyError("Missing 'timestamp ET'.")

    ts = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = ts

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


def fit_hmm(
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
# TRAIN-ONLY THRESHOLDS
# ============================================================


def calculate_thresholds(
    train: pd.DataFrame,
):

    thresholds = {}

    features = set()

    for hypothesis in HYPOTHESES:
        for feature, side in hypothesis["conditions"]:
            features.add(feature)

    for feature in features:
        values = (
            train[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        thresholds[(feature, "upper")] = float(values.quantile(0.80))

        thresholds[(feature, "lower")] = float(values.quantile(0.20))

    return thresholds


# ============================================================
# SIGNAL ENGINE
# ============================================================


def condition_passes(
    row,
    feature,
    side,
    threshold,
):

    value = row[feature]

    if pd.isna(value):
        return False

    if side == "upper":
        return value >= threshold

    if side == "lower":
        return value <= threshold

    raise ValueError(f"Unknown condition side: {side}")


def hypothesis_signal(
    row,
    hypothesis,
    thresholds,
):

    # --------------------------------------------------------
    # Regime is contextual, NOT directional.
    # --------------------------------------------------------

    if row["hmm_state"] != hypothesis["state"]:
        return False

    # --------------------------------------------------------
    # EVERY condition must pass.
    # --------------------------------------------------------

    for feature, side in hypothesis["conditions"]:
        threshold = thresholds[(feature, side)]

        if not condition_passes(
            row,
            feature,
            side,
            threshold,
        ):
            return False

    return True


# ============================================================
# BARRIER ENGINE
# ============================================================


def resolve_trade(
    session,
    entry_position,
    direction,
    target_points,
    stop_points,
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
        if direction == "long":
            target_hit = high[i] >= target

            stop_hit = low[i] <= stop

        else:
            target_hit = low[i] <= target

            stop_hit = high[i] >= stop

        # ----------------------------------------------------
        # Ambiguous OHLC candle.
        #
        # We use conservative treatment.
        # ----------------------------------------------------

        if target_hit and stop_hit:
            return {
                "exit_position": i,
                "pnl_points": -stop_points,
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
                "pnl_points": -stop_points,
                "reason": "stop",
            }

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

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
# EXECUTION SIMULATOR
# ============================================================


def run_hypothesis(
    validation,
    hypothesis,
    thresholds,
    target_points,
    barrier_name,
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

            # ------------------------------------------------
            # FLAT unless ALL conditions pass.
            # ------------------------------------------------

            signal = hypothesis_signal(
                row,
                hypothesis,
                thresholds,
            )

            if not signal:
                i += 1

                continue

            result = resolve_trade(
                session=session,
                entry_position=i,
                direction=hypothesis["direction"],
                target_points=target_points,
                stop_points=STOP_POINTS,
                horizon=horizon,
            )

            pnl_R = result["pnl_points"] / STOP_POINTS

            trades.append(
                {
                    "entry_timestamp": (positions[i]),
                    "exit_timestamp": (positions[result["exit_position"]]),
                    "hypothesis": (hypothesis["name"]),
                    "direction": (hypothesis["direction"]),
                    "state": (hypothesis["state"]),
                    "barrier": (barrier_name),
                    "horizon": horizon,
                    "pnl_points": (result["pnl_points"]),
                    "pnl_R": pnl_R,
                    "exit_reason": (result["reason"]),
                    "holding_bars": (result["exit_position"] - i),
                }
            )

            # ------------------------------------------------
            # CRITICAL:
            #
            # No overlapping trades.
            # ------------------------------------------------

            i = result["exit_position"] + 1

    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades,
):

    if trades.empty:
        return None

    pnl = trades["pnl_R"].astype(float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = -losses.sum()

    if gross_loss > 0:
        pf = gross_profit / gross_loss

    else:
        pf = np.inf

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    losing_streak = 0
    current_streak = 0

    for value in pnl:
        if value < 0:
            current_streak += 1

            losing_streak = max(
                losing_streak,
                current_streak,
            )

        else:
            current_streak = 0

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    daily_std = daily.std(ddof=1)

    if daily_std > 0 and len(daily) > 1:
        _daily_sharpe = daily.mean() / daily_std * np.sqrt(252)

    else:
        _daily_sharpe = np.nan

    return {
        "trades": len(trades),
        "win_rate": ((pnl > 0).mean()),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "profit_factor": pf,
        "max_drawdown_R": (drawdown.min()),
        "losing_streak": (losing_streak),
        "avg_holding_bars": (trades["holding_bars"].mean()),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("REAL CONDITIONAL DIRECTIONAL BARRIER TEST")

    print("=" * 110)

    print("\nThis test does NOT use HMM states as direct directional signals.")

    print("HMM = contextual regime.")

    print("Every hypothesis requires ALL directional conditions to agree.")

    print("No overlapping trades.")

    print("Thresholds are calculated from TRAIN only.")

    print("OOS parameters are never optimized.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    df = prepare_rth(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    print("\n=== ADDING DIRECTIONAL FEATURES ===")

    df = add_directional_features(df)

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    windows = generate_windows(df)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_results = []

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
        # HMM
        # ----------------------------------------------------

        train, validation = fit_hmm(
            train,
            validation,
        )

        # ----------------------------------------------------
        # TRAIN-ONLY THRESHOLDS
        # ----------------------------------------------------

        thresholds = calculate_thresholds(train)

        # ----------------------------------------------------
        # TEST COMPLETE HYPOTHESES
        # ----------------------------------------------------

        for hypothesis in HYPOTHESES:
            print(f"\nTesting: {hypothesis['name']}")

            for feature, side in hypothesis["conditions"]:
                threshold = thresholds[(feature, side)]

                print(f"  {feature:30s}{side:>8s}  threshold={threshold:.8f}")

            for barrier_name, target_points in BARRIERS:
                for horizon in HORIZONS:
                    trades = run_hypothesis(
                        validation=validation,
                        hypothesis=hypothesis,
                        thresholds=thresholds,
                        target_points=target_points,
                        barrier_name=barrier_name,
                        horizon=horizon,
                    )

                    result = calculate_metrics(trades)

                    if result is None:
                        continue

                    result.update(
                        {
                            "window": (window_number),
                            "hypothesis": (hypothesis["name"]),
                            "direction": (hypothesis["direction"]),
                            "barrier": (barrier_name),
                            "horizon": horizon,
                        }
                    )

                    all_results.append(result)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(all_results)

    if results.empty:
        print("\nNo trades generated.")

        return

    print("\n" + "=" * 110)

    print("COMPLETE HYPOTHESIS RESULTS")

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
            windows=(
                "window",
                "count",
            ),
            total_trades=(
                "trades",
                "sum",
            ),
            mean_WR=(
                "win_rate",
                "mean",
            ),
            median_R=(
                "mean_R",
                "median",
            ),
            mean_PF=(
                "profit_factor",
                "mean",
            ),
            median_PF=(
                "profit_factor",
                "median",
            ),
            total_R=(
                "total_R",
                "sum",
            ),
            worst_DD=(
                "max_drawdown_R",
                "min",
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

    # --------------------------------------------------------
    # POSITIVE CANDIDATES
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("POSITIVE MEDIAN OOS EXPECTANCY")

    print("=" * 110)

    positive = summary.loc[summary["median_R"] > 0]

    if positive.empty:
        print("NONE")

    else:
        print(positive.to_string(index=False))

    # --------------------------------------------------------
    # WINDOW CONSISTENCY
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("WINDOW-BY-WINDOW RESULTS")

    print("=" * 110)

    for _, candidate in positive.iterrows():
        mask = (
            (results["hypothesis"] == candidate["hypothesis"])
            & (results["barrier"] == candidate["barrier"])
            & (results["horizon"] == candidate["horizon"])
        )

        subset = results.loc[
            mask,
            [
                "window",
                "hypothesis",
                "direction",
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
    # SAVE
    # --------------------------------------------------------

    results.to_csv(
        "conditional_barrier_real_results.csv",
        index=False,
    )

    summary.to_csv(
        "conditional_barrier_real_summary.csv",
        index=False,
    )

    print("\nSaved:")

    print("conditional_barrier_real_results.csv")

    print("conditional_barrier_real_summary.csv")

    print("\n" + "=" * 110)

    print("TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
