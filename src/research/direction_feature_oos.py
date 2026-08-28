from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import (
    add_directional_features,
)

TRAIN_END = pd.Timestamp(
    "2024-12-31 16:59:00",
    tz="America/New_York",
)


MOMENTUM_FEATURES = [
    "directional_pressure_5",
    "directional_pressure_10",
    "directional_pressure_15",
    "directional_pressure_30",
    "direction_streak",
    "close_location_5",
    "close_location_10",
    "close_location_15",
    "close_location_30",
    "normalized_momentum_10",
    "normalized_momentum_15",
    "normalized_momentum_30",
]


FORWARD_HORIZONS = [
    5,
    15,
    30,
]


# ============================================================
# TIMESTAMP
# ============================================================


def get_timestamp_series(
    df: pd.DataFrame,
) -> pd.Series:

    if "timestamp ET" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp ET"],
            errors="coerce",
        )

        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("America/New_York")

        else:
            timestamps = timestamps.dt.tz_convert("America/New_York")

        return timestamps

    if isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        timestamps = pd.Series(
            df.index,
            index=df.index,
        )

        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("America/New_York")

        else:
            timestamps = timestamps.dt.tz_convert("America/New_York")

        return timestamps

    raise KeyError("Could not find timestamp ET column.")


# ============================================================
# QUANTILE BINS
# ============================================================


def calculate_train_quantile_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:

    values = (
        train[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    bins = values.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()

    bins = np.unique(bins)

    if len(bins) < 6:
        raise ValueError(
            f"Unable to construct five unique quantile bins for {feature}."
        )

    return bins


def assign_quantiles(
    df: pd.DataFrame,
    feature: str,
    bins: np.ndarray,
) -> pd.Series:

    values = df[feature]
    bin_edges = bins.tolist()

    return pd.cut(
        values,
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )


# ============================================================
# BOOTSTRAP
# ============================================================


def bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:

    rng = np.random.default_rng(seed)

    values_a = np.asarray(
        values_a,
        dtype=float,
    )

    values_b = np.asarray(
        values_b,
        dtype=float,
    )

    n_a = len(values_a)
    n_b = len(values_b)

    if n_a == 0 or n_b == 0:
        return np.nan, np.nan

    differences = np.empty(iterations)

    for i in range(iterations):
        sample_a = rng.choice(
            values_a,
            size=n_a,
            replace=True,
        )

        sample_b = rng.choice(
            values_b,
            size=n_b,
            replace=True,
        )

        differences[i] = sample_b.mean() - sample_a.mean()

    return (
        float(
            np.percentile(
                differences,
                2.5,
            )
        ),
        float(
            np.percentile(
                differences,
                97.5,
            )
        ),
    )


# ============================================================
# EXPECTANCY
# ============================================================


def calculate_expectancy_stats(
    returns: pd.Series,
) -> dict:

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(returns) == 0:
        return {
            "observations": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    positive = returns[returns > 0]

    negative = returns[returns < 0]

    gross_profit = positive.sum()

    gross_loss = abs(negative.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    else:
        profit_factor = np.inf

    return {
        "observations": len(returns),
        "mean": returns.mean(),
        "median": returns.median(),
        "win_rate": (returns.gt(0).mean()),
        "profit_factor": profit_factor,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("DIRECTIONAL FEATURE × HMM REGIME — OOS VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    df = add_directional_features(df)

    timestamps = get_timestamp_series(df)

    df = df.copy()

    df["_timestamp_et"] = timestamps

    # --------------------------------------------------------
    # RTH
    # --------------------------------------------------------

    if "market_period" in df.columns:
        rth = df.loc[df["market_period"] == "RTH"].copy()

    else:
        raise KeyError("market_period column is required.")

    # --------------------------------------------------------
    # TRAIN / OOS SPLIT
    # --------------------------------------------------------

    train = rth.loc[rth["_timestamp_et"] <= TRAIN_END].copy()

    oos = rth.loc[rth["_timestamp_et"] > TRAIN_END].copy()

    print("\n=== DATA SPLIT ===")

    print(
        "Train:",
        len(train),
    )

    print(
        "OOS:",
        len(oos),
    )

    print(
        "Train start:",
        train["_timestamp_et"].min(),
    )

    print(
        "Train end:",
        train["_timestamp_et"].max(),
    )

    print(
        "OOS start:",
        oos["_timestamp_et"].min(),
    )

    print(
        "OOS end:",
        oos["_timestamp_et"].max(),
    )

    # --------------------------------------------------------
    # FIT HMM ONLY ON TRAIN
    # --------------------------------------------------------

    print("\n=== FITTING HMM ===")

    model = VolatilityRegimeModel(
        n_states=3,
        random_state=42,
    )

    model.fit(train)

    train_states = model.predict_states(train)

    oos_states = model.predict_states(oos)

    train = train.copy()
    oos = oos.copy()

    train["hmm_state"] = train_states
    oos["hmm_state"] = oos_states

    print(
        "Converged:",
        model.model.monitor_.converged,
    )

    print(
        "Iterations:",
        model.model.monitor_.iter,
    )

    # --------------------------------------------------------
    # STATE PROPORTIONS
    # --------------------------------------------------------

    print("\nTrain regime proportions:")

    print(train["hmm_state"].value_counts(normalize=True).sort_index())

    print("\nOOS regime proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # --------------------------------------------------------
    # FEATURE TESTS
    # --------------------------------------------------------

    for feature in MOMENTUM_FEATURES:
        print("\n" + "#" * 70)

        print(f"FEATURE: {feature}")

        print("#" * 70)

        try:
            bins = calculate_train_quantile_bins(
                train,
                feature,
            )

        except ValueError as exc:
            print(f"SKIPPED: {exc}")

            continue

        print("\nTrain quantile boundaries:")

        for i, boundary in enumerate(bins):
            print(f"Boundary {i}: {boundary:.10f}")

        oos = oos.copy()

        oos["feature_quantile"] = assign_quantiles(
            oos,
            feature,
            bins,
        )

        # ----------------------------------------------------
        # HORIZONS
        # ----------------------------------------------------

        for horizon in FORWARD_HORIZONS:
            future_column = f"future_return_{horizon}"

            if future_column not in oos.columns:
                print(f"\nMissing {future_column}; skipping.")

                continue

            print("\n" + "-" * 70)

            print(f"{feature} -> {future_column}")

            print("-" * 70)

            # ------------------------------------------------
            # EACH HMM STATE
            # ------------------------------------------------

            for state in sorted(oos["hmm_state"].dropna().unique()):
                state_data = oos.loc[oos["hmm_state"] == state].copy()

                q1 = state_data.loc[
                    state_data["feature_quantile"] == 0,
                    future_column,
                ].dropna()

                q5 = state_data.loc[
                    state_data["feature_quantile"] == 4,
                    future_column,
                ].dropna()

                if len(q1) < 100 or len(q5) < 100:
                    print(f"\nSTATE {state}: insufficient observations")

                    continue

                # --------------------------------------------
                # LONG
                # --------------------------------------------

                long_q1 = calculate_expectancy_stats(q1)

                long_q5 = calculate_expectancy_stats(q5)

                long_difference = long_q5["mean"] - long_q1["mean"]

                long_ci = bootstrap_ci(
                    q1.to_numpy(),
                    q5.to_numpy(),
                    iterations=2000,
                    seed=42 + int(state),
                )

                # --------------------------------------------
                # SHORT
                #
                # For a short:
                #
                # P&L = - future return
                #
                # Therefore Q1 (most negative feature)
                # is the natural short-side condition.
                #
                # We compare:
                #
                # short Q1
                # versus
                # short Q5
                #
                # --------------------------------------------

                short_q1_returns = -q1
                short_q5_returns = -q5

                short_q1 = calculate_expectancy_stats(short_q1_returns)

                short_q5 = calculate_expectancy_stats(short_q5_returns)

                short_difference = short_q1["mean"] - short_q5["mean"]

                short_ci = bootstrap_ci(
                    short_q5_returns.to_numpy(),
                    short_q1_returns.to_numpy(),
                    iterations=2000,
                    seed=1000 + int(state),
                )

                print(f"\nSTATE {state}")

                print("\nLONG")

                print(f"Q1 observations: {long_q1['observations']}")

                print(f"Q5 observations: {long_q5['observations']}")

                print(f"Q1 mean: {long_q1['mean']:.10f}")

                print(f"Q5 mean: {long_q5['mean']:.10f}")

                print(f"Q5 - Q1: {long_difference:.10f}")

                print(f"95% bootstrap CI: [{long_ci[0]:.10f}, {long_ci[1]:.10f}]")

                print(f"Q5 LONG win rate: {long_q5['win_rate']:.4%}")

                print(f"Q5 LONG profit factor: {long_q5['profit_factor']:.4f}")

                if long_ci[0] > 0:
                    print("LONG RESULT: POSITIVE CONDITIONAL EFFECT")

                else:
                    print("LONG RESULT: CI CROSSES ZERO")

                print("\nSHORT")

                print(f"Q1 observations: {short_q1['observations']}")

                print(f"Q5 observations: {short_q5['observations']}")

                print(f"Q1 SHORT mean: {short_q1['mean']:.10f}")

                print(f"Q5 SHORT mean: {short_q5['mean']:.10f}")

                print(f"Q1 - Q5: {short_difference:.10f}")

                print(f"95% bootstrap CI: [{short_ci[0]:.10f}, {short_ci[1]:.10f}]")

                print(f"Q1 SHORT win rate: {short_q1['win_rate']:.4%}")

                print(f"Q1 SHORT profit factor: {short_q1['profit_factor']:.4f}")

                if short_ci[0] > 0:
                    print("SHORT RESULT: POSITIVE CONDITIONAL EFFECT")

                else:
                    print("SHORT RESULT: CI CROSSES ZERO")

    print("\n" + "=" * 70)

    print("DIRECTIONAL FEATURE OOS VALIDATION COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
