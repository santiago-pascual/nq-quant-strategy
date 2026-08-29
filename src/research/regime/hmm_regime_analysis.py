from __future__ import annotations

import pandas as pd

from src.data_loader import load_data
from src.models.regime import (
    HMM_FEATURES,
    VolatilityRegimeModel,
)


def main():

    df = load_data()

    model = VolatilityRegimeModel(
        n_states=3,
        random_state=42,
    )

    model.fit(df)

    states = model.predict_states(df)

    ("\n=== HMM SUMMARY ===")

    print("States:", model.n_states)
    print("Converged:", model.model.monitor_.converged)
    print("Iterations:", model.model.monitor_.iter)

    print("\nState counts:")
    print(states.value_counts().sort_index())

    print("\nTransition matrix:")
    print(
        pd.DataFrame(
            model.model.transmat_,
            index=[f"state_{i}" for i in range(model.n_states)],
            columns=[f"state_{i}" for i in range(model.n_states)],
        )
    )

    rth = df.loc[
        states.index,
        HMM_FEATURES
        + [
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ],
    ].copy()

    rth["hmm_state"] = states

    print("\n=== STATE CHARACTERISTICS ===")

    state_characteristics = rth.groupby("hmm_state")[
        HMM_FEATURES
        + [
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ]
    ].mean()

    pd.set_option(
        "display.max_columns",
        None,
    )

    print(state_characteristics)

    print("\nState proportions:")

    print(states.value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
