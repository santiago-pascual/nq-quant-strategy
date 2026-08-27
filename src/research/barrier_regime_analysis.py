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

HORIZONS = [5, 10, 15]

RISK_SCALES = [5, 10, 15, 20]

PAYOFF_MULTIPLES = {
    "0.50R": 0.50,
    "0.67R": 0.67,
    "0.75R": 0.75,
    "1.00R": 1.00,
}

DIRECTIONS = ["long", "short"]

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
# PREPARE RTH DATA
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

        if validation_end > end:
            validation_end = end

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
# FIRST BARRIER OUTCOME
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

                # With OHLC data, if both barriers
                # occur inside the same candle, the
                # order cannot be known.
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

    total = len(outcomes)

    if resolved == 0:
        return {
            "observations": total,
            "wins": 0,
            "losses": 0,
            "unresolved": unresolved,
            "resolution_rate": 0.0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
        }

    win_rate = wins / resolved

    gross_profit = wins * target_points

    gross_loss = losses * stop_points

    expectancy = (gross_profit - gross_loss) / resolved

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    return {
        "observations": total,
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolution_rate": (resolved / total),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


# ============================================================
# EVALUATE REGIME
# ============================================================


def evaluate_regime_configuration(
    validation: pd.DataFrame,
    direction: str,
    horizon: int,
    state: int,
    payoff_name: str,
    payoff_multiple: float,
    risk_points: float,
) -> dict:

    target_points = risk_points * payoff_multiple

    stop_points = risk_points

    regime_df = validation.loc[validation["hmm_state"] == state].copy()

    if regime_df.empty:
        return {
            "hmm_state": state,
            "observations": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "resolution_rate": np.nan,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
        }

    outcomes = calculate_barrier_outcomes(
        df=regime_df,
        target_points=target_points,
        stop_points=stop_points,
        horizon=horizon,
        direction=direction,
    )

    stats = calculate_statistics(
        outcomes=outcomes,
        target_points=target_points,
        stop_points=stop_points,
    )

    return {
        "hmm_state": state,
        "payoff_family": payoff_name,
        "payoff_multiple": payoff_multiple,
        "risk_points": risk_points,
        "target_points": target_points,
        "stop_points": stop_points,
        **stats,
    }


# ============================================================
# MAIN WALK-FORWARD EXPERIMENT
# ============================================================


def run_experiment(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    windows = generate_windows(rth)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_results = []

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

        if train.empty:
            continue

        if validation.empty:
            continue

        print(f"Train observations: {len(train)}")

        print(f"Validation observations: {len(validation)}")

        # ----------------------------------------------------
        # FIT HMM ON TRAIN ONLY
        # ----------------------------------------------------

        (
            train,
            validation,
            hmm,
        ) = fit_hmm(
            train,
            validation,
        )

        print(f"HMM converged: {hmm.model.monitor_.converged}")

        print("Validation regime proportions:")

        print(
            validation["hmm_state"]
            .value_counts(normalize=True)
            .sort_index()
            .to_string()
        )

        # ----------------------------------------------------
        # REGIME × DIRECTION × HORIZON
        # ----------------------------------------------------

        for horizon in HORIZONS:
            for direction in DIRECTIONS:
                for state in range(N_STATES):
                    for payoff_name, payoff_multiple in PAYOFF_MULTIPLES.items():
                        for risk_points in RISK_SCALES:
                            stats = evaluate_regime_configuration(
                                validation=validation,
                                direction=direction,
                                horizon=horizon,
                                state=state,
                                payoff_name=payoff_name,
                                payoff_multiple=payoff_multiple,
                                risk_points=risk_points,
                            )

                            all_results.append(
                                {
                                    "window": window_number,
                                    "validation_start": validation_start,
                                    "validation_end": validation_end,
                                    "direction": direction,
                                    "horizon": horizon,
                                    **stats,
                                }
                            )

    return pd.DataFrame(all_results)


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    valid = results.loc[results["observations"] > 0].copy()

    summary = (
        valid.groupby(
            [
                "direction",
                "horizon",
                "hmm_state",
                "payoff_family",
                "risk_points",
            ]
        )
        .agg(
            windows=(
                "window",
                "nunique",
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
            mean_win_rate=(
                "win_rate",
                "mean",
            ),
            mean_profit_factor=(
                "profit_factor",
                "mean",
            ),
            mean_resolution=(
                "resolution_rate",
                "mean",
            ),
            total_observations=(
                "observations",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["positive_window_rate"] = summary["positive_windows"] / summary["windows"]

    return summary


# ============================================================
# PRINT MAIN SUMMARY
# ============================================================


def print_main_summary(
    summary: pd.DataFrame,
) -> None:

    print("\n" + "=" * 120)

    print("REGIME × DIRECTION × PAYOFF — WALK-FORWARD SUMMARY")

    print("=" * 120)

    display = summary.copy()

    display["positive_window_rate"] = display["positive_window_rate"].map(
        lambda x: f"{x:.1%}"
    )

    display["mean_expectancy"] = display["mean_expectancy"].map(lambda x: f"{x:.4f}")

    display["median_expectancy"] = display["median_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    display["mean_win_rate"] = display["mean_win_rate"].map(lambda x: f"{x:.2%}")

    display["mean_profit_factor"] = display["mean_profit_factor"].map(
        lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
    )

    display["mean_resolution"] = display["mean_resolution"].map(lambda x: f"{x:.2%}")

    print(display.to_string(index=False))


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================


def build_robustness_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    robust = summary.loc[
        (summary["windows"] >= 3)
        & (summary["positive_window_rate"] >= 0.75)
        & (summary["median_expectancy"] > 0)
    ].copy()

    return robust


def print_robustness_summary(
    robust: pd.DataFrame,
) -> None:

    print("\n" + "=" * 120)

    print("ROBUST CONDITIONAL EDGES")

    print("=" * 120)

    if robust.empty:
        print("No configuration passed the diagnostic robustness filter.")

        print("This does NOT mean there is no edge.")

        print("It means the evidence is not yet stable enough under this filter.")

        return

    display = robust.copy()

    display = display.sort_values(
        [
            "direction",
            "horizon",
            "hmm_state",
            "median_expectancy",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    )

    display["positive_window_rate"] = display["positive_window_rate"].map(
        lambda x: f"{x:.1%}"
    )

    display["mean_expectancy"] = display["mean_expectancy"].map(lambda x: f"{x:.4f}")

    display["median_expectancy"] = display["median_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    display["mean_win_rate"] = display["mean_win_rate"].map(lambda x: f"{x:.2%}")

    display["mean_profit_factor"] = display["mean_profit_factor"].map(
        lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
    )

    print(
        display[
            [
                "direction",
                "horizon",
                "hmm_state",
                "payoff_family",
                "risk_points",
                "windows",
                "positive_window_rate",
                "median_expectancy",
                "mean_expectancy",
                "mean_win_rate",
                "mean_profit_factor",
                "total_observations",
            ]
        ].to_string(index=False)
    )


# ============================================================
# DIRECTION COMPARISON
# ============================================================


def print_direction_comparison(
    summary: pd.DataFrame,
) -> None:

    print("\n" + "=" * 120)

    print("LONG vs SHORT CONDITIONAL COMPARISON")

    print("=" * 120)

    comparison = (
        summary.groupby(
            [
                "direction",
                "horizon",
            ]
        )
        .agg(
            configurations=(
                "payoff_family",
                "count",
            ),
            positive_configurations=(
                "positive_window_rate",
                lambda x: (x >= 0.75).sum(),
            ),
            mean_expectancy=(
                "mean_expectancy",
                "mean",
            ),
            median_expectancy=(
                "median_expectancy",
                "median",
            ),
            mean_profit_factor=(
                "mean_profit_factor",
                "mean",
            ),
        )
        .reset_index()
    )

    comparison["mean_expectancy"] = comparison["mean_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    comparison["median_expectancy"] = comparison["median_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    comparison["mean_profit_factor"] = comparison["mean_profit_factor"].map(
        lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
    )

    print(comparison.to_string(index=False))


# ============================================================
# STATE COMPARISON
# ============================================================


def print_state_comparison(
    summary: pd.DataFrame,
) -> None:

    print("\n" + "=" * 120)

    print("HMM STATE COMPARISON")

    print("=" * 120)

    comparison = (
        summary.groupby(
            [
                "direction",
                "horizon",
                "hmm_state",
            ]
        )
        .agg(
            configurations=(
                "payoff_family",
                "count",
            ),
            positive_configurations=(
                "positive_window_rate",
                lambda x: (x >= 0.75).sum(),
            ),
            mean_expectancy=(
                "mean_expectancy",
                "mean",
            ),
            median_expectancy=(
                "median_expectancy",
                "median",
            ),
            mean_profit_factor=(
                "mean_profit_factor",
                "mean",
            ),
        )
        .reset_index()
    )

    comparison["mean_expectancy"] = comparison["mean_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    comparison["median_expectancy"] = comparison["median_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    comparison["mean_profit_factor"] = comparison["mean_profit_factor"].map(
        lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
    )

    print(comparison.to_string(index=False))


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 120)

    print("BARRIER × HMM REGIME × DIRECTION — OOS VALIDATION")

    print("=" * 120)

    print("\nThis experiment is diagnostic.")

    print("No final TP/SL is selected.")

    print("No ML model is trained.")

    print("No trading threshold is optimized.")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"RTH sessions: {rth['_session_id'].nunique()}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    results = run_experiment(rth)

    if results.empty:
        print("\nNo results generated.")

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = build_summary(results)

    print_main_summary(summary)

    # --------------------------------------------------------
    # ROBUST EDGES
    # --------------------------------------------------------

    robust = build_robustness_summary(summary)

    print_robustness_summary(robust)

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    print_direction_comparison(summary)

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    print_state_comparison(summary)

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 120)

    print("BARRIER REGIME ANALYSIS COMPLETE")

    print("=" * 120)

    print("\nInterpretation:")

    print("Positive expectancy alone is NOT sufficient.")

    print("Look for effects that persist across independent walk-forward windows.")

    print("Compare LONG vs SHORT.")

    print("Compare HMM states.")

    print("Compare payoff families.")

    print("Compare absolute barrier scales.")

    print("\nNo final strategy has been selected.")


if __name__ == "__main__":
    main()
