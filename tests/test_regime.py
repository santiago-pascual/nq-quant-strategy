import numpy as np
import pandas as pd

from src.models.regime import VolatilityRegimeModel


def make_test_data():

    return pd.DataFrame({
        "market_period": ["RTH"] * 4,
        "realized_vol_5": [1.0, 2.0, 3.0, 4.0],
        "realized_vol_15": [2.0, 4.0, 6.0, 8.0],
        "realized_vol_30": [3.0, 6.0, 9.0, 12.0],
        "realized_vol_60": [4.0, 8.0, 12.0, 16.0],
        "variance_ratio_5_30": [0.5, 1.0, 1.5, 2.0],
        "variance_ratio_5_60": [0.25, 0.5, 0.75, 1.0],
    })


def test_standardization():

    df = make_test_data()

    model = VolatilityRegimeModel()

    data = model.prepare_data(df)

    standardized = model.standardize(
        data,
        fit=True,
    )

    np.testing.assert_allclose(
        standardized.mean(),
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        standardized.std(),
        1.0,
        atol=1e-12,
    )


def test_probability_columns():

    model = VolatilityRegimeModel(
        n_states=3
    )

    probabilities = pd.DataFrame(
        {
            "state_probability_0": [0.7, 0.2],
            "state_probability_1": [0.2, 0.7],
            "state_probability_2": [0.1, 0.1],
        }
    )

    assert list(probabilities.columns) == [
        "state_probability_0",
        "state_probability_1",
        "state_probability_2",
    ]

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_next_state_probabilities():

    model = VolatilityRegimeModel(
        n_states=3
    )

    model.model.transmat_ = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8],
    ])

    current_probabilities = pd.DataFrame(
        {
            "state_probability_0": [1.0, 0.0, 0.0],
            "state_probability_1": [0.0, 1.0, 0.0],
            "state_probability_2": [0.0, 0.0, 1.0],
        }
    )

    next_probabilities = (
        model.next_state_probabilities(
            current_probabilities
        )
    )

    expected = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8],
    ])

    np.testing.assert_allclose(
        next_probabilities.to_numpy(),
        expected,
    )

    np.testing.assert_allclose(
        next_probabilities.sum(axis=1),
        1.0,
    )