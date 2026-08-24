from __future__ import annotations

import pandas as pd
from hmmlearn.hmm import GaussianHMM


HMM_FEATURES = [
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
]


class VolatilityRegimeModel:

    def __init__(
        self,
        n_states: int = 3,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.random_state = random_state

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )

        self.feature_means = None
        self.feature_stds = None

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = df.loc[
            df["market_period"] == "RTH",
            HMM_FEATURES,
        ].dropna()

        return data

    def standardize(
        self,
        data: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:

        if fit:
            self.feature_means = data.mean()
            self.feature_stds = data.std()

        standardized = (
            data - self.feature_means
        ) / self.feature_stds

        return standardized

    def fit(
        self,
        df: pd.DataFrame,
    ):

        data = self.prepare_data(df)

        standardized_data = self.standardize(
            data,
            fit=True,
        )

        self.model.fit(
            standardized_data
        )

        return self

    def predict_states(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        data = self.prepare_data(df)

        standardized_data = self.standardize(
            data,
            fit=False,
        )

        states = self.model.predict(
            standardized_data
        )

        return pd.Series(
            states,
            index=data.index,
            name="hmm_state",
        )

    def predict_probabilities(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = self.prepare_data(df)

        standardized_data = self.standardize(
            data,
            fit=False,
        )

        probabilities = (
            self.model.predict_proba(
                standardized_data
            )
        )

        columns = [
            f"state_probability_{i}"
            for i in range(self.n_states)
        ]

        return pd.DataFrame(
            probabilities,
            index=data.index,
            columns=columns,
        )

    def predict_regime_information(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        states = self.predict_states(df)

        probabilities = (
            self.predict_probabilities(df)
        )

        result = probabilities.copy()

        result["hmm_state"] = states

        return result

    def next_state_probabilities(
        self,
        current_probabilities: pd.DataFrame,
    ) -> pd.DataFrame:

        transition_matrix = self.model.transmat_

        next_probabilities = (
            current_probabilities
            .to_numpy()
            @ transition_matrix
        )

        columns = [
            f"next_state_probability_{i}"
            for i in range(self.n_states)
        ]

        return pd.DataFrame(
            next_probabilities,
            index=current_probabilities.index,
            columns=columns,
        )