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

    # ---------------------------------------------------------
    # 1. Train HMM only on historical training data
    # ---------------------------------------------------------

    model = VolatilityRegimeModel(
        n_states=3,
        random_state=42,
    )

    model.fit(train)

    # ---------------------------------------------------------
    # 2. Infer states on unseen 2025 data
    # ---------------------------------------------------------

    states = model.predict_states(oos)

    analysis = oos.loc[
        states.index,
        [
            "timestamp ET",
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ],
    ].copy()

    analysis["hmm_state"] = states

    # ---------------------------------------------------------
    # 3. Identify the next HMM state
    # ---------------------------------------------------------

    analysis["next_state"] = analysis["hmm_state"].shift(-1)

    # The final observation has no next state.
    analysis = analysis.dropna(subset=["next_state"])

    analysis["next_state"] = analysis["next_state"].astype(int)

    # ---------------------------------------------------------
    # 4. Analyze every state transition
    # ---------------------------------------------------------

    transition_stats = (
        analysis.groupby(["hmm_state", "next_state"])[
            [
                "future_vol_5",
                "future_vol_15",
                "future_vol_30",
            ]
        ]
        .agg(
            observations=("future_vol_5", "count"),
            mean_future_vol_5=("future_vol_5", "mean"),
            mean_future_vol_15=("future_vol_15", "mean"),
            mean_future_vol_30=("future_vol_30", "mean"),
        )
        .reset_index()
    )

    print("\n=== OOS HMM TRANSITION ANALYSIS ===")

    print(transition_stats.to_string(index=False))

    # ---------------------------------------------------------
    # 5. Transition frequencies
    # ---------------------------------------------------------

    transition_counts = pd.crosstab(
        analysis["hmm_state"],
        analysis["next_state"],
    )

    print("\n=== TRANSITION COUNTS ===")

    print(transition_counts)

    # ---------------------------------------------------------
    # 6. Transition probabilities observed in OOS
    # ---------------------------------------------------------

    transition_probabilities = transition_counts.div(
        transition_counts.sum(axis=1),
        axis=0,
    )

    print("\n=== OOS TRANSITION PROBABILITIES ===")

    print(transition_probabilities)


if __name__ == "__main__":
    main()
