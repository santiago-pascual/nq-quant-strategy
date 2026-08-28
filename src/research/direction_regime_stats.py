from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import (
    HMM_FEATURES,
    VolatilityRegimeModel,
)

MOMENTUM_FEATURES = [
    "past_return_10",
    "past_return_15",
    "past_return_30",
]

FUTURE_RETURN_TARGETS = [
    "future_return_5",
    "future_return_15",
    "future_return_30",
]

N_STATES = 3
N_QUANTILES = 5
RANDOM_STATE = 42

TRAIN_END = pd.Timestamp(
    "2024-12-31 16:59:00",
    tz="America/New_York",
)

OOS_START = pd.Timestamp(
    "2025-01-02 09:30:00",
    tz="America/New_York",
)


def assign_train_quantiles(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    feature: str,
):
    """
    Calculate quantile boundaries on TRAIN only and
    apply exactly those boundaries to TRAIN and OOS.
    """

    values = train[feature].dropna()

    bins = np.quantile(
        values,
        np.linspace(
            0,
            1,
            N_QUANTILES + 1,
        ),
    )

    bins = np.unique(bins)

    if len(bins) != N_QUANTILES + 1:
        raise ValueError(
            f"Could not create {N_QUANTILES} unique quantiles for {feature}."
        )

    bin_edges = bins.tolist()

    train_quantiles = pd.cut(
        train[feature],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    oos_quantiles = pd.cut(
        oos[feature],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    return (
        train_quantiles,
        oos_quantiles,
        bins,
    )


def bootstrap_mean_difference(
    q1: np.ndarray,
    q5: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
):
    """
    Bootstrap the difference:

        mean(Q5) - mean(Q1)

    Returns:
        observed_difference
        lower_95
        upper_95
    """

    rng = np.random.default_rng(seed)

    q1 = np.asarray(q1)
    q5 = np.asarray(q5)

    q1 = q1[np.isfinite(q1)]
    q5 = q5[np.isfinite(q5)]

    observed = q5.mean() - q1.mean()

    bootstrap_differences = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        q1_sample = rng.choice(
            q1,
            size=len(q1),
            replace=True,
        )

        q5_sample = rng.choice(
            q5,
            size=len(q5),
            replace=True,
        )

        bootstrap_differences[i] = q5_sample.mean() - q1_sample.mean()

    lower = np.percentile(
        bootstrap_differences,
        2.5,
    )

    upper = np.percentile(
        bootstrap_differences,
        97.5,
    )

    return (
        observed,
        lower,
        upper,
    )


def analyze_feature_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
):
    """
    Analyze Q5 versus Q1 conditional on HMM state.
    """

    print("\n" + "-" * 70)

    print(f"{feature} -> {target}")

    print("-" * 70)

    for state in sorted(df["hmm_state"].dropna().unique()):
        state_data = df.loc[df["hmm_state"] == state].copy()

        state_data = state_data.dropna(
            subset=[
                "momentum_quantile",
                target,
            ]
        )

        q1 = state_data.loc[
            state_data["momentum_quantile"] == 0,
            target,
        ].to_numpy()

        q5 = state_data.loc[
            state_data["momentum_quantile"] == 4,
            target,
        ].to_numpy()

        if len(q1) == 0 or len(q5) == 0:
            continue

        (
            difference,
            lower,
            upper,
        ) = bootstrap_mean_difference(
            q1,
            q5,
        )

        spearman = (
            state_data[
                [
                    "momentum_quantile",
                    target,
                ]
            ]
            .corr(method="spearman")
            .iloc[0, 1]
        )

        print(f"\nSTATE {int(state)}")

        print(f"Q1 observations: {len(q1)}")

        print(f"Q5 observations: {len(q5)}")

        print(f"Q1 mean: {q1.mean():.10f}")

        print(f"Q5 mean: {q5.mean():.10f}")

        print(f"Q5 - Q1: {difference:.10f}")

        print(f"95% bootstrap CI: [{lower:.10f}, {upper:.10f}]")

        print(f"Spearman: {spearman:.10f}")

        if lower > 0:
            print("RESULT: POSITIVE Q5-Q1 EFFECT")

        elif upper < 0:
            print("RESULT: NEGATIVE Q5-Q1 EFFECT")

        else:
            print("RESULT: CI CROSSES ZERO")


def main():

    print("=" * 70)
    print("DIRECTION × HMM REGIME — STATISTICAL VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required = (
        ["timestamp ET", "market_period"]
        + HMM_FEATURES
        + MOMENTUM_FEATURES
        + FUTURE_RETURN_TARGETS
    )

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = pd.to_datetime(df["timestamp ET"])

    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize("America/New_York")

    else:
        timestamp = timestamp.dt.tz_convert("America/New_York")

    # ========================================================
    # RTH
    # ========================================================

    rth_mask = df["market_period"] == "RTH"

    rth = df.loc[rth_mask].copy()

    rth_timestamp = timestamp.loc[rth.index]

    # ========================================================
    # TRAIN / OOS
    # ========================================================

    train = rth.loc[rth_timestamp <= TRAIN_END].copy()

    oos = rth.loc[rth_timestamp >= OOS_START].copy()

    print("\n=== DATA SPLIT ===")

    print(
        "Train:",
        len(train),
    )

    print(
        "OOS:",
        len(oos),
    )

    # ========================================================
    # HMM
    # ========================================================

    print("\n=== FITTING HMM ===")

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    print(
        "Converged:",
        model.model.monitor_.converged,
    )

    print(
        "Iterations:",
        model.model.monitor_.iter,
    )

    train["hmm_state"] = model.predict_states(train)

    oos["hmm_state"] = model.predict_states(oos)

    print("\nTrain regime proportions:")

    print(train["hmm_state"].value_counts(normalize=True).sort_index())

    print("\nOOS regime proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # ========================================================
    # DIRECTION × REGIME
    # ========================================================

    for feature in MOMENTUM_FEATURES:
        print("\n" + "#" * 70)

        print(f"FEATURE: {feature}")

        print("#" * 70)

        (
            train_quantiles,
            oos_quantiles,
            bins,
        ) = assign_train_quantiles(
            train,
            oos,
            feature,
        )

        train["momentum_quantile"] = train_quantiles

        oos["momentum_quantile"] = oos_quantiles

        print("\nTrain quantile boundaries:")

        for i, value in enumerate(bins):
            print(f"Boundary {i}: {value:.10f}")

        for target in FUTURE_RETURN_TARGETS:
            analyze_feature_target(
                oos,
                feature,
                target,
            )

    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
