from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

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
        n_iter: int = 200,
    ):

        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter

        self.scaler = StandardScaler()

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )

        self._is_fitted = False

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        missing_features = [
            feature for feature in HMM_FEATURES if feature not in df.columns
        ]

        if missing_features:
            raise KeyError(f"Missing HMM features: {missing_features}")

        data = df[HMM_FEATURES].copy()

        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        data = data.dropna()

        if data.empty:
            raise ValueError("No valid observations available for HMM.")

        return data

    def standardize(
        self,
        data: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:

        if fit:
            self.scaler.fit(data)

            standardized = self.scaler.transform(data)

            standardized = standardized * np.sqrt((len(data) - 1) / len(data))

        else:
            standardized = self.scaler.transform(data)

        return pd.DataFrame(
            standardized,
            index=data.index,
            columns=data.columns,
        )

    def fit(
        self,
        df: pd.DataFrame,
    ) -> VolatilityRegimeModel:

        data = self.prepare_data(df)

        standardized = self.standardize(
            data,
            fit=True,
        )

        self.model.fit(standardized.to_numpy())

        self._is_fitted = True

        return self

    def predict_states(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting states.")

        data = self.prepare_data(df)

        standardized = self.standardize(
            data,
            fit=False,
        )

        states = self.model.predict(standardized.to_numpy())

        return pd.Series(
            states,
            index=data.index,
            name="hmm_state",
        )

    def predict_probabilities(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting probabilities.")

        data = self.prepare_data(df)

        standardized = self.standardize(
            data,
            fit=False,
        )

        probabilities = self.model.predict_proba(standardized.to_numpy())

        columns = [f"state_probability_{i}" for i in range(self.n_states)]

        return pd.DataFrame(
            probabilities,
            index=data.index,
            columns=columns,
        )

    def next_state_probabilities(
        self,
        current_probabilities: pd.DataFrame,
    ) -> pd.DataFrame:

        expected_columns = [f"state_probability_{i}" for i in range(self.n_states)]

        missing_columns = [
            column
            for column in expected_columns
            if column not in current_probabilities.columns
        ]

        if missing_columns:
            raise KeyError(f"Missing probability columns: {missing_columns}")

        current = current_probabilities[expected_columns].to_numpy()

        next_probabilities = current @ self.model.transmat_

        return pd.DataFrame(
            next_probabilities,
            index=current_probabilities.index,
            columns=expected_columns,
        )

    def predict_next_state_probabilities(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        probabilities = self.predict_probabilities(df)

        return self.next_state_probabilities(probabilities)
