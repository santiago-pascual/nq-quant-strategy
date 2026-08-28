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

TRAIN_END = pd.Timestamp(
    "2024-12-31 16:59:00",
    tz="America/New_York",
)

OOS_START = pd.Timestamp(
    "2025-01-02 09:30:00",
    tz="America/New_York",
)


# ============================================================
# QUANTILES
# ============================================================


def calculate_train_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:
    """
    Calculate quantile boundaries using TRAIN data only.
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


def apply_quantiles(
    df: pd.DataFrame,
    feature: str,
    bins: np.ndarray,
) -> pd.Series:
    """
    Apply TRAIN-derived quantile boundaries.
    """

    bin_edges = bins.tolist()

    return pd.cut(
        df[feature],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )


# ============================================================
# EXPECTANCY
# ============================================================


def calculate_expectancy(
    returns: pd.Series,
) -> dict:
    """
    Calculate directional expectancy statistics.

    EV = P(win) * AvgWin - P(loss) * AvgLoss
    """

    values = returns.dropna().to_numpy()

    if len(values) == 0:
        return {
            "observations": 0,
            "win_rate": np.nan,
            "loss_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "mean_return": np.nan,
            "median_return": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
        }

    wins = values[values > 0]
    losses = values[values < 0]

    win_rate = len(wins) / len(values)

    loss_rate = len(losses) / len(values)

    avg_win = wins.mean() if len(wins) > 0 else 0.0

    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

    expectancy = win_rate * avg_win - loss_rate * avg_loss

    gross_profit = wins.sum() if len(wins) > 0 else 0.0

    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "observations": len(values),
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "mean_return": values.mean(),
        "median_return": np.median(values),
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


# ============================================================
# ANALYSIS
# ============================================================


def analyze_condition(
    df: pd.DataFrame,
    state: int,
    quantile: int,
    target: str,
) -> dict:
    """
    Calculate expectancy for one
    HMM-state × momentum-quantile condition.
    """

    subset = df.loc[
        (df["hmm_state"] == state) & (df["momentum_quantile"] == quantile),
        target,
    ]

    result = calculate_expectancy(subset)

    result["state"] = state
    result["quantile"] = quantile
    result["target"] = target

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("DIRECTIONAL EXPECTANCY — OOS VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    required_columns = (
        [
            "timestamp ET",
            "market_period",
        ]
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

    print("\n=== RTH DATA ===")

    print(
        "Observations:",
        len(rth),
    )

    # ========================================================
    # TRAIN / OOS
    # ========================================================

    train = rth.loc[rth_timestamp <= TRAIN_END].copy()

    oos = rth.loc[rth_timestamp >= OOS_START].copy()

    print("\n=== DATA SPLIT ===")

    print(
        "Train observations:",
        len(train),
    )

    print(
        "OOS observations:",
        len(oos),
    )

    print(
        "Train end:",
        TRAIN_END,
    )

    print(
        "OOS start:",
        OOS_START,
    )

    # ========================================================
    # FIT HMM ON TRAIN ONLY
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

    # ========================================================
    # PREDICT HMM STATES
    # ========================================================

    train["hmm_state"] = model.predict_states(train)

    oos["hmm_state"] = model.predict_states(oos)

    print("\nOOS regime proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    all_results = []

    for feature in MOMENTUM_FEATURES:
        print("\n" + "#" * 70)

        print(f"MOMENTUM FEATURE: {feature}")

        print("#" * 70)

        # ----------------------------------------------------
        # TRAIN-ONLY BINS
        # ----------------------------------------------------

        bins = calculate_train_bins(
            train,
            feature,
        )

        print("\nTrain quantile boundaries:")

        for i, boundary in enumerate(bins):
            print(f"Boundary {i}: {boundary:.10f}")

        train["momentum_quantile"] = apply_quantiles(
            train,
            feature,
            bins,
        )

        oos["momentum_quantile"] = apply_quantiles(
            oos,
            feature,
            bins,
        )

        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        for target in FUTURE_RETURN_TARGETS:
            print("\n" + "-" * 70)

            print(f"{feature} -> {target}")

            print("-" * 70)

            # ------------------------------------------------
            # FULL STATE × QUANTILE TABLE
            # ------------------------------------------------

            for state in range(N_STATES):
                print(f"\nSTATE {state}")

                state_results = []

                for quantile in range(N_QUANTILES):
                    result = analyze_condition(
                        oos,
                        state,
                        quantile,
                        target,
                    )

                    all_results.append(result)

                    state_results.append(result)

                    print(f"\nQ{quantile + 1}")

                    print(f"Observations: {result['observations']}")

                    print(f"Win rate: {result['win_rate']:.4f}")

                    print(f"Avg win: {result['avg_win']:.10f}")

                    print(f"Avg loss: {result['avg_loss']:.10f}")

                    print(f"Mean return: {result['mean_return']:.10f}")

                    print(f"Median return: {result['median_return']:.10f}")

                    print(f"Expectancy: {result['expectancy']:.10f}")

                    print(f"Profit factor: {result['profit_factor']:.4f}")

                # ------------------------------------------------
                # Q5 VS Q1 EXPECTANCY
                # ------------------------------------------------

                q1 = state_results[0]
                q5 = state_results[4]

                expectancy_difference = q5["expectancy"] - q1["expectancy"]

                print("\nQ5 - Q1 EXPECTANCY:")

                print(f"{expectancy_difference:.10f}")

                print(f"Q5 EXPECTANCY: {q5['expectancy']:.10f}")

                print(f"Q5 PROFIT FACTOR: {q5['profit_factor']:.4f}")

                print(f"Q5 WIN RATE: {q5['win_rate']:.4f}")

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("Q5 EXPECTANCY SUMMARY")
    print("=" * 70)

    summary = results_df.loc[
        results_df["quantile"] == 4,
        [
            "state",
            "target",
            "observations",
            "win_rate",
            "mean_return",
            "expectancy",
            "profit_factor",
        ],
    ].copy()

    print(summary.to_string(index=False))

    # ========================================================
    # POSITIVE EXPECTANCY CONDITIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("POSITIVE EXPECTANCY CONDITIONS")
    print("=" * 70)

    positive = summary.loc[summary["expectancy"] > 0]

    if positive.empty:
        print("No positive expectancy Q5 conditions found.")

    else:
        print(positive.to_string(index=False))

    print("\n" + "=" * 70)
    print("EXPECTANCY VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
