from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import HMM_FEATURES, VolatilityRegimeModel


# ============================================================
# CONFIGURATION
# ============================================================

MOMENTUM_FEATURES = [
    "past_return_10",
    "past_return_15",
    "past_return_30",
]

TARGETS = {
    5: "future_return_5",
    15: "future_return_15",
    30: "future_return_30",
}

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
# QUANTILE FUNCTIONS
# ============================================================


def calculate_train_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:

    values = train[feature].dropna()

    if values.empty:
        raise ValueError(f"No valid training observations for {feature}.")

    bins = np.quantile(
        values,
        np.linspace(0.0, 1.0, N_QUANTILES + 1),
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

    return pd.cut(
        df[feature],
        bins=bins,
        labels=False,
        include_lowest=True,
    )


# ============================================================
# EXPECTANCY
# ============================================================


def calculate_statistics(
    returns: pd.Series,
) -> dict:

    values = returns.dropna().to_numpy()

    if len(values) == 0:
        return {
            "observations": 0,
            "win_rate": np.nan,
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

    avg_win = wins.mean() if len(wins) else 0.0

    avg_loss = abs(losses.mean()) if len(losses) else 0.0

    expectancy = win_rate * avg_win - loss_rate * avg_loss

    gross_profit = wins.sum() if len(wins) else 0.0

    gross_loss = abs(losses.sum()) if len(losses) else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "observations": len(values),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "mean_return": values.mean(),
        "median_return": np.median(values),
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


# ============================================================
# NON-OVERLAPPING SAMPLING
# ============================================================


def create_nonoverlapping_sample(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Select observations spaced by the forward-return horizon.

    Example:
        horizon = 15

        observations:
        0, 15, 30, 45, ...

    Therefore the forward return windows do not overlap.
    """

    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    sampled_parts = []

    for offset in range(horizon):
        part = df.iloc[offset::horizon].copy()

        if not part.empty:
            sampled_parts.append(part)

    if not sampled_parts:
        return df.iloc[0:0].copy()

    # Use a single deterministic stream.
    #
    # We deliberately select the first offset only.
    # The remaining offsets are reserved for robustness
    # checks and are not pooled into the primary result.
    return sampled_parts[0]


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("DIRECTION × HMM — NON-OVERLAPPING OOS VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    required = (
        [
            "timestamp ET",
            "market_period",
        ]
        + HMM_FEATURES
        + MOMENTUM_FEATURES
        + list(TARGETS.values())
    )

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

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
    print("Observations:", len(rth))

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

    print("\n=== OOS STATE PROPORTIONS ===")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # ========================================================
    # ANALYSIS
    # ========================================================

    summary_rows = []

    for feature in MOMENTUM_FEATURES:
        print("\n" + "#" * 70)
        print(f"FEATURE: {feature}")
        print("#" * 70)

        bins = calculate_train_bins(
            train,
            feature,
        )

        print("\nTrain quantile boundaries:")

        for i, boundary in enumerate(bins):
            print(f"Boundary {i}: {boundary:.10f}")

        oos["momentum_quantile"] = apply_quantiles(
            oos,
            feature,
            bins,
        )

        for horizon, target in TARGETS.items():
            print("\n" + "-" * 70)

            print(f"{feature} -> {target} (NON-OVERLAPPING)")

            print("-" * 70)

            sampled = create_nonoverlapping_sample(
                oos,
                horizon,
            )

            print(
                "Sampled observations:",
                len(sampled),
            )

            # ------------------------------------------------
            # STATES
            # ------------------------------------------------

            for state in range(N_STATES):
                print(f"\nSTATE {state}")

                state_results = {}

                for quantile in range(N_QUANTILES):
                    subset = sampled.loc[
                        (sampled["hmm_state"] == state)
                        & (sampled["momentum_quantile"] == quantile),
                        target,
                    ]

                    stats = calculate_statistics(subset)

                    state_results[quantile] = stats

                    print(f"\nQ{quantile + 1}")

                    print(
                        "Observations:",
                        stats["observations"],
                    )

                    print(
                        "Win rate:",
                        f"{stats['win_rate']:.4f}",
                    )

                    print(
                        "Mean return:",
                        f"{stats['mean_return']:.10f}",
                    )

                    print(
                        "Expectancy:",
                        f"{stats['expectancy']:.10f}",
                    )

                    print(
                        "Profit factor:",
                        f"{stats['profit_factor']:.4f}",
                    )

                # ------------------------------------------------
                # Q5 VS Q1
                # ------------------------------------------------

                q1 = state_results[0]
                q5 = state_results[4]

                if np.isfinite(q1["expectancy"]) and np.isfinite(q5["expectancy"]):
                    q5_q1 = q5["expectancy"] - q1["expectancy"]

                else:
                    q5_q1 = np.nan

                print(
                    "\nQ5 - Q1 EXPECTANCY:",
                    f"{q5_q1:.10f}",
                )

                summary_rows.append(
                    {
                        "feature": feature,
                        "horizon": horizon,
                        "state": state,
                        "q1_expectancy": q1["expectancy"],
                        "q5_expectancy": q5["expectancy"],
                        "q5_minus_q1": q5_q1,
                        "q5_win_rate": q5["win_rate"],
                        "q5_profit_factor": q5["profit_factor"],
                        "q5_observations": q5["observations"],
                    }
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 70)
    print("NON-OVERLAPPING Q5 SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    # ========================================================
    # STRONGEST CONDITIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("POSITIVE Q5 EXPECTANCY CONDITIONS")
    print("=" * 70)

    positive = summary.loc[
        (summary["q5_expectancy"] > 0) & (summary["q5_profit_factor"] > 1.0)
    ].sort_values(
        "q5_minus_q1",
        ascending=False,
    )

    if positive.empty:
        print("No positive Q5 conditions.")

    else:
        print(positive.to_string(index=False))

    print("\n" + "=" * 70)
    print("NON-OVERLAPPING VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
