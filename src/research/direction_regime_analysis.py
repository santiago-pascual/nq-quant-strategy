from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import (
    HMM_FEATURES,
    VolatilityRegimeModel,
)


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# QUANTILE FUNCTIONS
# ============================================================


def calculate_train_quantile_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:
    """
    Calculate momentum quantile boundaries using TRAIN data only.
    """

    values = train[feature].dropna()

    if values.empty:
        raise ValueError(f"No valid training observations for {feature}.")

    bins = np.quantile(
        values,
        np.linspace(
            0.0,
            1.0,
            N_QUANTILES + 1,
        ),
    )

    bins = np.unique(bins)

    if len(bins) != N_QUANTILES + 1:
        raise ValueError(
            f"Could not create {N_QUANTILES} unique quantile bins for {feature}."
        )

    return bins


def assign_quantiles(
    df: pd.DataFrame,
    feature: str,
    bins: np.ndarray,
) -> pd.Series:
    """
    Assign observations to momentum quantiles using
    boundaries calculated from TRAIN data.
    """

    return pd.cut(
        df[feature],
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )


# ============================================================
# REGIME × DIRECTION ANALYSIS
# ============================================================


def analyze_regime_momentum(
    df: pd.DataFrame,
    feature: str,
    future_return: str,
) -> None:
    """
    Analyze the relationship between momentum and future
    return conditional on HMM regime.
    """

    print("\n" + "-" * 70)
    print(f"{feature} -> {future_return}")
    print("-" * 70)

    states = sorted(df["hmm_state"].dropna().unique())

    for state in states:
        state_data = df.loc[df["hmm_state"] == state].copy()

        state_data = state_data.dropna(
            subset=[
                "momentum_quantile",
                future_return,
            ]
        )

        if state_data.empty:
            continue

        grouped = state_data.groupby(
            "momentum_quantile",
            observed=False,
        )[future_return].agg(
            observations="count",
            mean="mean",
            median="median",
        )

        print(f"\nHMM STATE {int(state)}")

        print(grouped)

        # ----------------------------------------------------
        # Q5 - Q1
        # ----------------------------------------------------

        if len(grouped) >= 2:
            q1_mean = grouped.iloc[0]["mean"]
            q5_mean = grouped.iloc[-1]["mean"]

            q5_minus_q1 = q5_mean - q1_mean

            print(f"\nQ5 - Q1 mean return: {q5_minus_q1:.10f}")

            if q1_mean != 0:
                print(f"Q5 / Q1 mean ratio: {q5_mean / q1_mean:.4f}")

        # ----------------------------------------------------
        # SPEARMAN
        # ----------------------------------------------------

        correlation_data = state_data[
            [
                "momentum_quantile",
                future_return,
            ]
        ].dropna()

        if len(correlation_data) >= 2:
            spearman = correlation_data["momentum_quantile"].corr(
                correlation_data[future_return],
                method="spearman",
            )

            print(f"Spearman correlation: {spearman:.10f}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("DIRECTION × HMM REGIME ANALYSIS")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_data()

    # load_data() already provides:
    #
    # - return features
    # - momentum features
    # - volatility features
    # - future volatility targets
    # - future return targets
    # - market_period
    # - timestamp ET

    # ========================================================
    # 2. VERIFY REQUIRED COLUMNS
    # ========================================================

    required_columns = (
        ["timestamp ET", "market_period"]
        + HMM_FEATURES
        + MOMENTUM_FEATURES
        + FUTURE_RETURN_TARGETS
    )

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # ========================================================
    # 3. PARSE TIMESTAMP
    # ========================================================

    timestamp_et = pd.to_datetime(
        df["timestamp ET"],
    )

    if timestamp_et.dt.tz is None:
        timestamp_et = timestamp_et.dt.tz_localize("America/New_York")

    else:
        timestamp_et = timestamp_et.dt.tz_convert("America/New_York")

    # ========================================================
    # 4. RTH DATA
    # ========================================================

    rth_mask = df["market_period"] == "RTH"

    rth = df.loc[rth_mask].copy()

    rth_timestamp = timestamp_et.loc[rth.index]

    print("\n=== RTH DATA ===")

    print(
        "Observations:",
        len(rth),
    )

    if rth.empty:
        raise ValueError("No RTH observations available.")

    # ========================================================
    # 5. TRAIN / OOS SPLIT
    # ========================================================

    train_end = pd.Timestamp(
        "2024-12-31 16:59:00",
        tz="America/New_York",
    )

    oos_start = pd.Timestamp(
        "2025-01-02 09:30:00",
        tz="America/New_York",
    )

    train_mask = rth_timestamp <= train_end

    oos_mask = rth_timestamp >= oos_start

    train = rth.loc[train_mask].copy()

    oos = rth.loc[oos_mask].copy()

    train_timestamp = rth_timestamp.loc[train.index]

    oos_timestamp = rth_timestamp.loc[oos.index]

    print("\n=== DATA SPLIT ===")

    print(
        "Train observations:",
        len(train),
    )

    print(
        "Train start:",
        train_timestamp.min(),
    )

    print(
        "Train end:",
        train_timestamp.max(),
    )

    print(
        "\nOOS observations:",
        len(oos),
    )

    print(
        "OOS start:",
        oos_timestamp.min(),
    )

    print(
        "OOS end:",
        oos_timestamp.max(),
    )

    if train.empty:
        raise ValueError("Training dataset is empty.")

    if oos.empty:
        raise ValueError("OOS dataset is empty.")

    # ========================================================
    # 6. FIT HMM ON TRAIN ONLY
    # ========================================================

    print("\n" + "=" * 70)
    print("FITTING HMM ON TRAIN DATA")
    print("=" * 70)

    hmm_model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    hmm_model.fit(train)

    print(
        "Converged:",
        hmm_model.model.monitor_.converged,
    )

    print(
        "Iterations:",
        hmm_model.model.monitor_.iter,
    )

    # ========================================================
    # 7. PREDICT STATES
    # ========================================================

    train_states = hmm_model.predict_states(train)

    oos_states = hmm_model.predict_states(oos)

    train["hmm_state"] = train_states

    oos["hmm_state"] = oos_states

    # ========================================================
    # 8. REGIME PROPORTIONS
    # ========================================================

    print("\n=== TRAIN REGIME PROPORTIONS ===")

    print(train["hmm_state"].value_counts(normalize=True).sort_index())

    print("\n=== OOS REGIME PROPORTIONS ===")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # ========================================================
    # 9. DIRECTION × REGIME
    # ========================================================

    for feature in MOMENTUM_FEATURES:
        print("\n" + "#" * 70)

        print(f"MOMENTUM FEATURE: {feature}")

        print("#" * 70)

        # ----------------------------------------------------
        # TRAIN-ONLY QUANTILE BOUNDARIES
        # ----------------------------------------------------

        bins = calculate_train_quantile_bins(
            train=train,
            feature=feature,
        )

        print("\nTrain quantile boundaries:")

        for i, boundary in enumerate(bins):
            print(f"Boundary {i}: {boundary:.10f}")

        # ----------------------------------------------------
        # APPLY TRAIN BINS
        # ----------------------------------------------------

        train["momentum_quantile"] = assign_quantiles(
            train,
            feature,
            bins,
        )

        oos["momentum_quantile"] = assign_quantiles(
            oos,
            feature,
            bins,
        )

        # ----------------------------------------------------
        # OOS ANALYSIS
        # ----------------------------------------------------

        for future_return in FUTURE_RETURN_TARGETS:
            analyze_regime_momentum(
                df=oos,
                feature=feature,
                future_return=future_return,
            )

    # ========================================================
    # 10. HMM TRANSITION MATRIX
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINED HMM TRANSITION MATRIX")
    print("=" * 70)

    transition_matrix = pd.DataFrame(
        hmm_model.model.transmat_,
        index=[f"state_{i}" for i in range(N_STATES)],
        columns=[f"state_{i}" for i in range(N_STATES)],
    )

    print(transition_matrix)


if __name__ == "__main__":
    main()
