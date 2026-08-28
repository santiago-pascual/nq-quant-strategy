from __future__ import annotations

import pandas as pd

from src.data_loader import load_data
from src.models.regime import (
    HMM_FEATURES,
    VolatilityRegimeModel,
)

TRAIN_END = "2024-12-31"
OOS_START = "2025-01-01"


def split_train_oos(df: pd.DataFrame):

    train = df[df["timestamp ET"] < TRAIN_END].copy()

    oos = df[df["timestamp ET"] >= OOS_START].copy()

    return train, oos


def summarize_states(
    df: pd.DataFrame,
    states: pd.Series,
) -> pd.DataFrame:

    data = df.loc[
        states.index,
        HMM_FEATURES
        + [
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ],
    ].copy()

    data["hmm_state"] = states

    return data.groupby("hmm_state")[
        HMM_FEATURES
        + [
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ]
    ].mean()


def main():

    df = load_data()

    train, oos = split_train_oos(df)

    print("\n=== DATA SPLIT ===")
    print("Train:", train["timestamp ET"].min())
    print("Train end:", train["timestamp ET"].max())
    print("OOS:", oos["timestamp ET"].min())
    print("OOS end:", oos["timestamp ET"].max())

    model = VolatilityRegimeModel(
        n_states=3,
        random_state=42,
    )

    # ---------------------------------------------------------
    # 1. Fit ONLY on training data
    # ---------------------------------------------------------

    model.fit(train)

    print("\n=== TRAINING MODEL ===")
    print(
        "Converged:",
        model.model.monitor_.converged,
    )
    print(
        "Iterations:",
        model.model.monitor_.iter,
    )

    # ---------------------------------------------------------
    # 2. Infer states on train and OOS
    # ---------------------------------------------------------

    train_states = model.predict_states(train)
    oos_states = model.predict_states(oos)

    # ---------------------------------------------------------
    # 3. Compare state proportions
    # ---------------------------------------------------------

    print("\n=== TRAIN STATE PROPORTIONS ===")

    print(train_states.value_counts(normalize=True).sort_index())

    print("\n=== OOS STATE PROPORTIONS ===")

    print(oos_states.value_counts(normalize=True).sort_index())

    # ---------------------------------------------------------
    # 4. State characteristics
    # ---------------------------------------------------------

    print("\n=== TRAIN STATE CHARACTERISTICS ===")

    print(
        summarize_states(
            train,
            train_states,
        )
    )

    print("\n=== OOS STATE CHARACTERISTICS ===")

    print(
        summarize_states(
            oos,
            oos_states,
        )
    )

    # ---------------------------------------------------------
    # 5. Transition matrix learned from TRAIN
    # ---------------------------------------------------------

    print("\n=== TRAINED TRANSITION MATRIX ===")

    print(
        pd.DataFrame(
            model.model.transmat_,
            index=[f"state_{i}" for i in range(model.n_states)],
            columns=[f"state_{i}" for i in range(model.n_states)],
        )
    )


if __name__ == "__main__":
    main()
