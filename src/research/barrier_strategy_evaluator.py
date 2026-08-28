from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_YEARS = 2
VALIDATION_MONTHS = 3

N_STATES = 3
RANDOM_STATE = 42

HORIZONS = [5, 10, 15]

RISK_SCALES = [5, 10, 15, 20]

PAYOFF_MULTIPLES = {
    "0.50R": 0.50,
    "0.67R": 0.67,
    "0.75R": 0.75,
    "1.00R": 1.00,
}

DIRECTIONS = [
    "long",
    "short",
]

# ------------------------------------------------------------
# CONDITIONAL EDGE FILTER
#
# These are intentionally broad diagnostic thresholds.
# They are NOT optimized thresholds.
#
# A configuration is allowed to trade when:
#
#   1. Its historical conditional expectancy from the
#      training period is positive.
#
#   2. The corresponding HMM regime is known.
#
# We do NOT use validation results to decide whether to trade.
# ------------------------------------------------------------

MIN_TRAIN_EXPECTANCY = 0.0

# Minimum number of resolved training observations required
# before considering a configuration.
MIN_TRAIN_RESOLVED = 100


# ============================================================
# TIMESTAMP
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


# ============================================================
# PREPARE RTH
# ============================================================


def prepare_rth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_timestamp_et"] = get_timestamp_series(df)

    if "market_period" not in df.columns:
        raise KeyError("Missing 'market_period' column.")

    rth = df.loc[df["market_period"] == "RTH"].copy()

    rth = rth.sort_values("_timestamp_et")

    rth = rth.set_index("_timestamp_et")

    rth.index.name = "timestamp_et"

    if "session_date" in rth.columns:
        rth["_session_id"] = rth["session_date"].astype(str)
    else:
        rth["_session_id"] = rth.index.date

    return rth


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    rth: pd.DataFrame,
) -> list[
    tuple[
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:

    start = rth.index.min()
    end = rth.index.max()

    validation_start = start + pd.DateOffset(years=TRAIN_YEARS)

    windows = []

    while validation_start < end:
        train_start = validation_start - pd.DateOffset(years=TRAIN_YEARS)

        validation_end = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)

        validation_end = min(validation_end, end)

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


def fit_hmm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    VolatilityRegimeModel,
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
        model,
    )


# ============================================================
# BARRIER OUTCOME
# ============================================================


def first_barrier_outcome(
    session: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    n = len(session)

    outcomes = np.zeros(
        n,
        dtype=np.int8,
    )

    for i in range(n):
        end = min(
            i + horizon + 1,
            n,
        )

        if i + 1 >= end:
            continue

        entry = close[i]

        future_high = high[i + 1 : end]

        future_low = low[i + 1 : end]

        if direction == "long":
            target_price = entry + target_points

            stop_price = entry - stop_points

            for bar_high, bar_low in zip(
                future_high,
                future_low,
            ):
                target_hit = bar_high >= target_price

                stop_hit = bar_low <= stop_price

                # We cannot know which barrier was
                # touched first from OHLC alone.
                if target_hit and stop_hit:
                    outcomes[i] = 0
                    break

                if target_hit:
                    outcomes[i] = 1
                    break

                if stop_hit:
                    outcomes[i] = -1
                    break

        elif direction == "short":
            target_price = entry - target_points

            stop_price = entry + stop_points

            for bar_high, bar_low in zip(
                future_high,
                future_low,
            ):
                target_hit = bar_low <= target_price

                stop_hit = bar_high >= stop_price

                if target_hit and stop_hit:
                    outcomes[i] = 0
                    break

                if target_hit:
                    outcomes[i] = 1
                    break

                if stop_hit:
                    outcomes[i] = -1
                    break

        else:
            raise ValueError("direction must be 'long' or 'short'.")

    return pd.Series(
        outcomes,
        index=session.index,
        dtype="int8",
    )


# ============================================================
# SESSION-AWARE BARRIER ENGINE
# ============================================================


def calculate_barrier_outcomes(
    df: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:

    result = pd.Series(
        0,
        index=df.index,
        dtype="int8",
    )

    for _, session in df.groupby(
        "_session_id",
        sort=False,
    ):
        session_result = first_barrier_outcome(
            session=session,
            target_points=target_points,
            stop_points=stop_points,
            horizon=horizon,
            direction=direction,
        )

        result.loc[session_result.index] = session_result

    return result


# ============================================================
# TRAINING EXPECTANCY
# ============================================================


def calculate_training_expectancy(
    outcomes: pd.Series,
    target_points: float,
    stop_points: float,
) -> tuple[float, int]:

    wins = int((outcomes == 1).sum())

    losses = int((outcomes == -1).sum())

    resolved = wins + losses

    if resolved == 0:
        return (
            np.nan,
            0,
        )

    expectancy = (wins * target_points - losses * stop_points) / resolved

    return (
        float(expectancy),
        resolved,
    )


# ============================================================
# BUILD TRAINING EDGE MAP
# ============================================================


def build_training_edge_map(
    train: pd.DataFrame,
) -> dict:

    edge_map = {}

    for direction in DIRECTIONS:
        for horizon in HORIZONS:
            for state in range(N_STATES):
                state_df = train.loc[train["hmm_state"] == state].copy()

                for payoff_name, payoff_multiple in PAYOFF_MULTIPLES.items():
                    for risk_points in RISK_SCALES:
                        target_points = risk_points * payoff_multiple

                        stop_points = risk_points

                        if state_df.empty:
                            edge_map[
                                (
                                    direction,
                                    horizon,
                                    state,
                                    payoff_name,
                                    risk_points,
                                )
                            ] = {
                                "expectancy": np.nan,
                                "resolved": 0,
                            }

                            continue

                        outcomes = calculate_barrier_outcomes(
                            df=state_df,
                            target_points=target_points,
                            stop_points=stop_points,
                            horizon=horizon,
                            direction=direction,
                        )

                        expectancy, resolved = calculate_training_expectancy(
                            outcomes=outcomes,
                            target_points=target_points,
                            stop_points=stop_points,
                        )

                        edge_map[
                            (
                                direction,
                                horizon,
                                state,
                                payoff_name,
                                risk_points,
                            )
                        ] = {
                            "expectancy": expectancy,
                            "resolved": resolved,
                        }

    return edge_map


# ============================================================
# VALIDATION TRADE GENERATOR
# ============================================================


def generate_validation_trades(
    validation: pd.DataFrame,
    edge_map: dict,
) -> pd.DataFrame:

    rows = []

    # --------------------------------------------------------
    # Calculate every possible candidate outcome first.
    #
    # This means the final decision is made using only
    # information available at the beginning of each bar.
    # --------------------------------------------------------

    for direction in DIRECTIONS:
        for horizon in HORIZONS:
            for state in range(N_STATES):
                for payoff_name, payoff_multiple in PAYOFF_MULTIPLES.items():
                    for risk_points in RISK_SCALES:
                        key = (
                            direction,
                            horizon,
                            state,
                            payoff_name,
                            risk_points,
                        )

                        training_info = edge_map[key]

                        train_expectancy = training_info["expectancy"]

                        train_resolved = training_info["resolved"]

                        # ------------------------------------------------
                        # Only configurations that had positive
                        # training expectancy are eligible.
                        # ------------------------------------------------

                        if not np.isfinite(train_expectancy):
                            continue

                        if train_expectancy <= MIN_TRAIN_EXPECTANCY:
                            continue

                        if train_resolved < MIN_TRAIN_RESOLVED:
                            continue

                        regime_df = validation.loc[
                            validation["hmm_state"] == state
                        ].copy()

                        if regime_df.empty:
                            continue

                        target_points = risk_points * payoff_multiple

                        stop_points = risk_points

                        outcomes = calculate_barrier_outcomes(
                            df=regime_df,
                            target_points=target_points,
                            stop_points=stop_points,
                            horizon=horizon,
                            direction=direction,
                        )

                        for timestamp, outcome in outcomes.items():
                            if outcome == 0:
                                continue

                            if outcome == 1:
                                pnl_points = target_points
                            else:
                                pnl_points = -stop_points

                            rows.append(
                                {
                                    "timestamp": timestamp,
                                    "direction": direction,
                                    "horizon": horizon,
                                    "hmm_state": state,
                                    "payoff_family": payoff_name,
                                    "risk_points": risk_points,
                                    "target_points": target_points,
                                    "stop_points": stop_points,
                                    "train_expectancy": train_expectancy,
                                    "train_resolved": train_resolved,
                                    "outcome": int(outcome),
                                    "pnl_points": pnl_points,
                                }
                            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# BUILD COMBINED LONG / SHORT / FLAT
# ============================================================


def build_combined_strategy(
    candidate_trades: pd.DataFrame,
) -> pd.DataFrame:

    if candidate_trades.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # We need one decision per timestamp.
    #
    # If multiple LONG configurations agree, their average
    # predicted/training expectancy is used.
    #
    # Same for SHORT.
    #
    # Then:
    #
    #   LONG expectancy > SHORT expectancy and > 0
    #       -> LONG
    #
    #   SHORT expectancy > LONG expectancy and > 0
    #       -> SHORT
    #
    #   otherwise
    #       -> FLAT
    #
    # This is deliberately simple and diagnostic.
    # It is NOT an optimized meta-model.
    # --------------------------------------------------------

    grouped = candidate_trades.groupby(
        [
            "timestamp",
            "direction",
        ],
        as_index=False,
    ).agg(
        mean_train_expectancy=(
            "train_expectancy",
            "mean",
        ),
        best_train_expectancy=(
            "train_expectancy",
            "max",
        ),
        n_candidates=(
            "train_expectancy",
            "count",
        ),
    )

    pivot = grouped.pivot(
        index="timestamp",
        columns="direction",
        values="best_train_expectancy",
    ).reset_index()

    if "long" not in pivot.columns:
        pivot["long"] = np.nan

    if "short" not in pivot.columns:
        pivot["short"] = np.nan

    pivot["long"] = pivot["long"].fillna(-np.inf)

    pivot["short"] = pivot["short"].fillna(-np.inf)

    pivot["decision"] = "flat"

    long_condition = (pivot["long"] > 0) & (pivot["long"] > pivot["short"])

    short_condition = (pivot["short"] > 0) & (pivot["short"] > pivot["long"])

    pivot.loc[
        long_condition,
        "decision",
    ] = "long"

    pivot.loc[
        short_condition,
        "decision",
    ] = "short"

    # --------------------------------------------------------
    # For each timestamp and selected direction, select the
    # configuration with the highest TRAINING expectancy.
    #
    # Again: selection is based on training only.
    # --------------------------------------------------------

    selected_rows = []

    for _, row in pivot.iterrows():
        timestamp = row["timestamp"]
        decision = row["decision"]

        if decision == "flat":
            selected_rows.append(
                {
                    "timestamp": timestamp,
                    "decision": "flat",
                    "pnl_points": 0.0,
                    "risk_points": np.nan,
                    "payoff_family": None,
                    "horizon": np.nan,
                    "hmm_state": np.nan,
                }
            )
            continue

        candidates = candidate_trades.loc[
            (candidate_trades["timestamp"] == timestamp)
            & (candidate_trades["direction"] == decision)
        ].copy()

        if candidates.empty:
            selected_rows.append(
                {
                    "timestamp": timestamp,
                    "decision": "flat",
                    "pnl_points": 0.0,
                    "risk_points": np.nan,
                    "payoff_family": None,
                    "horizon": np.nan,
                    "hmm_state": np.nan,
                }
            )
            continue

        selected = candidates.sort_values(
            "train_expectancy",
            ascending=False,
        ).iloc[0]

        selected_rows.append(
            {
                "timestamp": timestamp,
                "decision": decision,
                "pnl_points": selected["pnl_points"],
                "risk_points": selected["risk_points"],
                "payoff_family": selected["payoff_family"],
                "horizon": selected["horizon"],
                "hmm_state": selected["hmm_state"],
            }
        )

    return pd.DataFrame(selected_rows)


# ============================================================
# METRICS
# ============================================================


def calculate_equity_metrics(
    trades: pd.DataFrame,
) -> dict[str, float]:

    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "total_pnl_points": 0.0,
            "mean_pnl_points": np.nan,
            "profit_factor": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown_points": 0.0,
            "max_drawdown_pct": np.nan,
            "longest_losing_streak": 0,
            "profitable_days": 0,
            "losing_days": 0,
        }

    pnl = trades["pnl_points"].astype(float)

    wins = int((pnl > 0).sum())

    losses = int((pnl < 0).sum())

    resolved = wins + losses

    if resolved > 0:
        win_rate = wins / resolved
    else:
        win_rate = np.nan

    total_pnl = pnl.sum()

    mean_pnl = pnl.mean()

    gross_profit = pnl[pnl > 0].sum()

    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    # --------------------------------------------------------
    # Trade-level Sharpe
    # --------------------------------------------------------

    std = pnl.std(ddof=1)

    if std > 0 and len(pnl) > 1:
        sharpe = pnl.mean() / std * np.sqrt(len(pnl))
    else:
        sharpe = np.nan

    # --------------------------------------------------------
    # Sortino
    # --------------------------------------------------------

    downside = pnl[pnl < 0]

    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan

    if np.isfinite(downside_std) and downside_std > 0:
        sortino = pnl.mean() / downside_std * np.sqrt(len(pnl))
    else:
        sortino = np.nan

    # --------------------------------------------------------
    # Equity / drawdown
    # --------------------------------------------------------

    equity = pnl.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = drawdown.min()

    if equity.max() != 0:
        max_drawdown_pct = max_drawdown / equity.max()
    else:
        max_drawdown_pct = np.nan

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
    # Daily P&L
    # --------------------------------------------------------

    daily = trades.set_index("timestamp")["pnl_points"].resample("1D").sum()

    profitable_days = int((daily > 0).sum())

    losing_days = int((daily < 0).sum())

    return {
        "trades": len(pnl),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl_points": total_pnl,
        "mean_pnl_points": mean_pnl,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_points": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "longest_losing_streak": longest_losing_streak,
        "profitable_days": profitable_days,
        "losing_days": losing_days,
    }


# ============================================================
# WINDOW EVALUATION
# ============================================================


def evaluate_window(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    window_number: int,
) -> dict:

    train, validation, _hmm = fit_hmm(
        train,
        validation,
    )

    edge_map = build_training_edge_map(train)

    candidates = generate_validation_trades(
        validation=validation,
        edge_map=edge_map,
    )

    if candidates.empty:
        return {
            "window": window_number,
            "candidate_trades": pd.DataFrame(),
            "combined_trades": pd.DataFrame(),
        }

    combined = build_combined_strategy(candidates)

    # --------------------------------------------------------
    # Remove FLAT rows from trade statistics.
    # --------------------------------------------------------

    executed = combined.loc[combined["decision"] != "flat"].copy()

    return {
        "window": window_number,
        "candidate_trades": candidates,
        "combined_trades": executed,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================


def main():

    print("=" * 110)

    print("BARRIER STRATEGY EVALUATOR — LONG / SHORT / FLAT")

    print("=" * 110)

    print("\nDiagnostic experiment only.")

    print("No final parameters are being selected.")

    print("No XGBoost model is being trained.")

    print("No transaction costs or slippage are applied yet.")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"RTH sessions: {rth['_session_id'].nunique()}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    windows = generate_windows(rth)

    print(f"\nWalk-forward windows: {len(windows)}")

    # --------------------------------------------------------
    # COLLECT RESULTS
    # --------------------------------------------------------

    all_window_trades = []

    window_metrics = []

    for window_number, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        print("\n" + "#" * 100)

        print(f"WINDOW {window_number}")

        print("#" * 100)

        train = rth.loc[
            (rth.index >= train_start) & (rth.index < validation_start)
        ].copy()

        validation = rth.loc[
            (rth.index >= validation_start) & (rth.index < validation_end)
        ].copy()

        print(f"Train: {train_start.date()} → {validation_start.date()}")

        print(f"Validation: {validation_start.date()} → {validation_end.date()}")

        print(f"Train observations: {len(train)}")

        print(f"Validation observations: {len(validation)}")

        result = evaluate_window(
            train=train,
            validation=validation,
            window_number=window_number,
        )

        trades = result["combined_trades"]

        if trades.empty:
            print("No trades generated.")

            continue

        trades = trades.copy()

        trades["window"] = window_number

        all_window_trades.append(trades)

        metrics = calculate_equity_metrics(trades)

        metrics["window"] = window_number

        window_metrics.append(metrics)

        print("\nWINDOW RESULTS")

        for key, value in metrics.items():
            if key == "window":
                continue

            if isinstance(
                value,
                float,
            ):
                if np.isfinite(value):
                    print(f"{key:30s}: {value:.6f}")
                else:
                    print(f"{key:30s}: nan")

            else:
                print(f"{key:30s}: {value}")

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not all_window_trades:
        print("\nNo executable trades were generated.")

        return

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    combined_trades = pd.concat(
        all_window_trades,
        ignore_index=True,
    )

    combined_trades = combined_trades.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall = calculate_equity_metrics(combined_trades)

    print("\n" + "=" * 110)

    print("COMBINED OOS RESULTS — LONG / SHORT / FLAT")

    print("=" * 110)

    for key, value in overall.items():
        if isinstance(
            value,
            float,
        ):
            if np.isfinite(value):
                print(f"{key:30s}: {value:.6f}")
            else:
                print(f"{key:30s}: nan")

        else:
            print(f"{key:30s}: {value}")

    # --------------------------------------------------------
    # DIRECTION BREAKDOWN
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("DIRECTION BREAKDOWN")

    print("=" * 110)

    for direction in [
        "long",
        "short",
    ]:
        direction_trades = combined_trades.loc[combined_trades["decision"] == direction]

        metrics = calculate_equity_metrics(direction_trades)

        print(f"\n{direction.upper()}")

        for key, value in metrics.items():
            print(f"{key:30s}: {value}")

    # --------------------------------------------------------
    # HMM STATE BREAKDOWN
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("HMM STATE BREAKDOWN")

    print("=" * 110)

    for state in range(N_STATES):
        state_trades = combined_trades.loc[combined_trades["hmm_state"] == state]

        metrics = calculate_equity_metrics(state_trades)

        print(f"\nSTATE {state}")

        for key, value in metrics.items():
            print(f"{key:30s}: {value}")

    # --------------------------------------------------------
    # PAYOFF BREAKDOWN
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("PAYOFF BREAKDOWN")

    print("=" * 110)

    payoff_summary = (
        combined_trades.groupby("payoff_family")
        .agg(
            trades=(
                "pnl_points",
                "count",
            ),
            total_pnl=(
                "pnl_points",
                "sum",
            ),
            mean_pnl=(
                "pnl_points",
                "mean",
            ),
            win_rate=(
                "pnl_points",
                lambda x: (x > 0).mean(),
            ),
        )
        .reset_index()
    )

    print(payoff_summary.to_string(index=False))

    # --------------------------------------------------------
    # LONG / SHORT DECISION COUNTS
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("DECISION COUNTS")

    print("=" * 110)

    print(combined_trades["decision"].value_counts().to_string())

    # --------------------------------------------------------
    # SAVE DIAGNOSTIC OUTPUT
    # --------------------------------------------------------

    output_path = "barrier_strategy_evaluator_results.csv"

    combined_trades.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved trade-level results to:")

    print(output_path)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("BARRIER STRATEGY EVALUATION COMPLETE")

    print("=" * 110)

    print("\nIMPORTANT:")

    print("This is still research.")

    print("The decision layer is intentionally simple.")

    print("No threshold optimization has been performed.")

    print("No costs or slippage have been included.")

    print("No position sizing has been included.")

    print("No funded-account constraints have been included.")

    print("Do NOT treat the resulting P&L as live-trading performance.")


if __name__ == "__main__":
    main()
