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

HORIZONS = [5, 10, 15, 30]

TARGETS = [5, 10, 15, 20]

STOPS = [5, 10, 15, 20]

N_STATES = 3

RANDOM_STATE = 42


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
# RTH DATA
# ============================================================


def prepare_rth_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_timestamp_et"] = get_timestamp_series(df)

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
# FIRST BARRIER
# ============================================================


def first_barrier_outcome(
    session: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:
    """
    +1 = target first
    -1 = stop first
     0 = unresolved / ambiguous
    """

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
    )


# ============================================================
# SESSION-AWARE BARRIERS
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
            session,
            target_points,
            stop_points,
            horizon,
            direction,
        )

        result.loc[session_result.index] = session_result

    return result


# ============================================================
# STATISTICS
# ============================================================


def calculate_statistics(
    outcomes: pd.Series,
    target_points: float,
    stop_points: float,
) -> dict[str, float]:

    wins = int((outcomes == 1).sum())

    losses = int((outcomes == -1).sum())

    unresolved = int((outcomes == 0).sum())

    resolved = wins + losses

    if resolved == 0:
        return {
            "wins": wins,
            "losses": losses,
            "unresolved": unresolved,
            "resolution_rate": 0.0,
            "win_rate": np.nan,
            "expectancy": np.nan,
        }

    win_rate = wins / resolved

    expectancy = ((wins * target_points) - (losses * stop_points)) / resolved

    return {
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolution_rate": (resolved / len(outcomes)),
        "win_rate": win_rate,
        "expectancy": expectancy,
    }


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

    hmm = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    hmm.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = hmm.predict_states(train)

    validation["hmm_state"] = hmm.predict_states(validation)

    return (
        train,
        validation,
        hmm,
    )


# ============================================================
# PARAMETER EVALUATION
# ============================================================


def evaluate_parameters(
    df: pd.DataFrame,
    direction: str,
    horizon: int,
) -> pd.DataFrame:

    rows = []

    for target in TARGETS:
        for stop in STOPS:
            outcomes = calculate_barrier_outcomes(
                df,
                target,
                stop,
                horizon,
                direction,
            )

            stats = calculate_statistics(
                outcomes,
                target,
                stop,
            )

            rows.append(
                {
                    "direction": direction,
                    "horizon": horizon,
                    "target": target,
                    "stop": stop,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    rth: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate:

        train_start
        validation_start
        validation_end

    using rolling calendar windows.
    """

    start = rth.index.min()

    end = rth.index.max()

    windows = []

    validation_start = start + pd.DateOffset(years=TRAIN_YEARS)

    while validation_start < end:
        validation_end = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)

        if validation_end > end:
            validation_end = end

        train_start = validation_start - pd.DateOffset(years=TRAIN_YEARS)

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
# SELECT PARAMETER ON TRAIN
# ============================================================


def select_best_parameters(
    train: pd.DataFrame,
    direction: str,
    horizon: int,
) -> tuple[int, int]:
    """
    Select barrier parameters using TRAIN ONLY.

    Selection criterion:

        highest raw expectancy

    subject to:

        resolution >= 50%
        at least 10,000 resolved observations
    """

    grid = evaluate_parameters(
        train,
        direction,
        horizon,
    )

    candidates = grid.loc[
        (grid["resolution_rate"] >= 0.50)
        & ((grid["wins"] + grid["losses"]) >= 10_000)
        & grid["expectancy"].notna()
    ].copy()

    if candidates.empty:
        return (
            10,
            15,
        )

    best = candidates.sort_values(
        "expectancy",
        ascending=False,
    ).iloc[0]

    return (
        int(best["target"]),
        int(best["stop"]),
    )


# ============================================================
# VALIDATE SELECTED PARAMETERS
# ============================================================


def validate_parameters(
    validation: pd.DataFrame,
    direction: str,
    horizon: int,
    target: int,
    stop: int,
) -> dict[str, float]:

    outcomes = calculate_barrier_outcomes(
        validation,
        target,
        stop,
        horizon,
        direction,
    )

    stats = calculate_statistics(
        outcomes,
        target,
        stop,
    )

    return {
        "direction": direction,
        "horizon": horizon,
        "target": target,
        "stop": stop,
        **stats,
    }


# ============================================================
# PRINT WALK-FORWARD RESULTS
# ============================================================


def print_results(
    results: pd.DataFrame,
) -> None:

    if results.empty:
        print("No walk-forward results.")
        return

    print("\n" + "=" * 80)

    print("WALK-FORWARD RESULTS")

    print("=" * 80)

    display = results.copy()

    display["resolution_rate"] = display["resolution_rate"].map(lambda x: f"{x:.2%}")

    display["win_rate"] = display["win_rate"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else "nan"
    )

    display["expectancy"] = display["expectancy"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "nan"
    )

    print(display.to_string(index=False))


# ============================================================
# STABILITY SUMMARY
# ============================================================


def print_stability_summary(
    results: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)

    print("WALK-FORWARD STABILITY SUMMARY")

    print("=" * 80)

    for direction in [
        "long",
        "short",
    ]:
        for horizon in HORIZONS:
            subset = results.loc[
                (results["direction"] == direction) & (results["horizon"] == horizon)
            ]

            if subset.empty:
                continue

            positive_rate = (subset["expectancy"] > 0).mean()

            median_expectancy = subset["expectancy"].median()

            mean_expectancy = subset["expectancy"].mean()

            print(f"\n{direction.upper()} {horizon}-bar")

            print(f"Windows: {len(subset)}")

            print(f"Positive expectancy windows: {positive_rate:.2%}")

            print(f"Median expectancy: {median_expectancy:.4f}")

            print(f"Mean expectancy: {mean_expectancy:.4f}")


# ============================================================
# REGIME VALIDATION
# ============================================================


def validate_regime_stability(
    validation: pd.DataFrame,
    direction: str,
    horizon: int,
    target: int,
    stop: int,
) -> pd.DataFrame:

    rows = []

    for state, regime_df in validation.groupby(
        "hmm_state",
        sort=True,
    ):
        outcomes = calculate_barrier_outcomes(
            regime_df,
            target,
            stop,
            horizon,
            direction,
        )

        stats = calculate_statistics(
            outcomes,
            target,
            stop,
        )

        rows.append(
            {
                "hmm_state": state,
                "direction": direction,
                "horizon": horizon,
                "target": target,
                "stop": stop,
                **stats,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 80)

    print("WALK-FORWARD BARRIER VALIDATION")

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth_data(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"RTH sessions: {rth['_session_id'].nunique()}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    windows = generate_windows(rth)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_results = []

    regime_results = []

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
        print("\n" + "#" * 80)

        print(f"WINDOW {window_number}")

        print("#" * 80)

        print(f"Train: {train_start} → {validation_start}")

        print(f"Validation: {validation_start} → {validation_end}")

        train = rth.loc[
            (rth.index >= train_start) & (rth.index < validation_start)
        ].copy()

        validation = rth.loc[
            (rth.index >= validation_start) & (rth.index < validation_end)
        ].copy()

        if len(train) == 0:
            continue

        if len(validation) == 0:
            continue

        print(f"Train observations: {len(train)}")

        print(f"Validation observations: {len(validation)}")

        # ----------------------------------------------------
        # HMM FIT ONLY ON TRAIN
        # ----------------------------------------------------

        (
            train_hmm,
            validation_hmm,
            hmm,
        ) = fit_hmm(
            train,
            validation,
        )

        print(f"HMM converged: {hmm.model.monitor_.converged}")

        # ----------------------------------------------------
        # EACH HORIZON
        # ----------------------------------------------------

        for horizon in HORIZONS:
            for direction in [
                "long",
                "short",
            ]:
                # --------------------------------------------
                # SELECT ON TRAIN
                # --------------------------------------------

                target, stop = select_best_parameters(
                    train_hmm,
                    direction,
                    horizon,
                )

                # --------------------------------------------
                # TEST ON VALIDATION
                # --------------------------------------------

                stats = validate_parameters(
                    validation_hmm,
                    direction,
                    horizon,
                    target,
                    stop,
                )

                stats["window"] = window_number

                stats["train_start"] = train_start

                stats["validation_start"] = validation_start

                stats["validation_end"] = validation_end

                all_results.append(stats)

                print(
                    f"\n"
                    f"{direction.upper()} "
                    f"{horizon}-bar | "
                    f"Selected "
                    f"{target}/{stop} | "
                    f"Validation "
                    f"WR="
                    f"{stats['win_rate']:.2%} | "
                    f"Exp="
                    f"{stats['expectancy']:.4f}"
                )

                # --------------------------------------------
                # REGIME CONDITIONAL
                # --------------------------------------------

                regime_stats = validate_regime_stability(
                    validation_hmm,
                    direction,
                    horizon,
                    target,
                    stop,
                )

                regime_stats["window"] = window_number

                regime_results.append(regime_stats)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(all_results)

    print_results(results)

    print_stability_summary(results)

    # --------------------------------------------------------
    # REGIME RESULTS
    # --------------------------------------------------------

    if regime_results:
        regime_df = pd.concat(
            regime_results,
            ignore_index=True,
        )

        print("\n" + "=" * 80)

        print("REGIME-CONDITIONAL WALK-FORWARD RESULTS")

        print("=" * 80)

        regime_display = regime_df.copy()

        regime_display["resolution_rate"] = regime_display["resolution_rate"].map(
            lambda x: f"{x:.2%}"
        )

        regime_display["win_rate"] = regime_display["win_rate"].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else "nan"
        )

        regime_display["expectancy"] = regime_display["expectancy"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "nan"
        )

        print(regime_display.to_string(index=False))

        # ----------------------------------------------------
        # REGIME SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 80)

        print("REGIME STABILITY SUMMARY")

        print("=" * 80)

        summary = (
            regime_df.groupby(
                [
                    "direction",
                    "horizon",
                    "hmm_state",
                ]
            )
            .agg(
                windows=(
                    "window",
                    "count",
                ),
                positive_windows=(
                    "expectancy",
                    lambda x: (x > 0).sum(),
                ),
                mean_expectancy=(
                    "expectancy",
                    "mean",
                ),
                median_expectancy=(
                    "expectancy",
                    "median",
                ),
            )
            .reset_index()
        )

        summary["positive_rate"] = summary["positive_windows"] / summary["windows"]

        print(summary.to_string(index=False))

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("WALK-FORWARD VALIDATION COMPLETE")

    print("=" * 80)

    print("\nIMPORTANT:")

    print("Barrier parameters were selected using TRAIN data only.")

    print("Validation results were not used to select parameters.")

    print("No final trading rule has been selected.")

    print("No transaction costs or funded-account execution rules")

    print("have been applied yet.")


if __name__ == "__main__":
    main()
