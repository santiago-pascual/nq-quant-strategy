from __future__ import annotations

import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel

TRAIN_END = "2024-12-31"
OOS_START = "2025-01-01"


def split_train_oos(df: pd.DataFrame):

    train = df[df["timestamp ET"] < TRAIN_END].copy()

    oos = df[df["timestamp ET"] >= OOS_START].copy()

    return train, oos


def main():

    df = load_data()

    train, oos = split_train_oos(df)

    model = VolatilityRegimeModel(
        n_states=3,
        random_state=42,
    )

    # Train ONLY on historical data.
    model.fit(train)

    # Get live-compatible regime information.
    regime = model.predict_regime_information(oos)

    next_regime = model.next_state_probabilities(
        regime[
            [
                "state_probability_0",
                "state_probability_1",
                "state_probability_2",
            ]
        ]
    )

    result = pd.concat(
        [regime, next_regime],
        axis=1,
    )

    print("\n=== OOS REGIME PROBABILITY CHECK ===")

    print("Rows:", len(result))

    print("\nProbability sums:")

    print(
        result[
            [
                "state_probability_0",
                "state_probability_1",
                "state_probability_2",
            ]
        ]
        .sum(axis=1)
        .describe()
    )

    print("\nNext-state probability sums:")

    print(
        result[
            [
                "next_state_probability_0",
                "next_state_probability_1",
                "next_state_probability_2",
            ]
        ]
        .sum(axis=1)
        .describe()
    )

    print("\nFirst 10 observations:")

    print(result.head(10).to_string())

    print("\nMaximum state probability:")

    print(
        result[
            [
                "state_probability_0",
                "state_probability_1",
                "state_probability_2",
            ]
        ]
        .max(axis=1)
        .describe()
    )


if __name__ == "__main__":
    main()
