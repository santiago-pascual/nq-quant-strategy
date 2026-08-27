import numpy as np
import pandas as pd

from src.targets import (
    add_future_return_targets,
    add_future_volatility_targets,
)


def test_future_volatility_targets():

    log_returns = [0.01] * 35

    df = pd.DataFrame({"log_return": log_returns})

    df = add_future_volatility_targets(df)

    expected_vol_5 = np.sqrt(5 * 0.01**2)
    expected_vol_15 = np.sqrt(15 * 0.01**2)
    expected_vol_30 = np.sqrt(30 * 0.01**2)

    np.testing.assert_allclose(
        df["future_vol_5"].iloc[0],
        expected_vol_5,
    )

    np.testing.assert_allclose(
        df["future_vol_15"].iloc[0],
        expected_vol_15,
    )

    np.testing.assert_allclose(
        df["future_vol_30"].iloc[0],
        expected_vol_30,
    )


def test_future_targets_do_not_use_past_data():

    df = pd.DataFrame(
        {
            "log_return": [
                0.01,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.01,
                0.01,
                0.01,
                0.01,
            ]
        }
    )

    df = add_future_volatility_targets(df)

    expected = np.sqrt(5 * 0.50**2)

    np.testing.assert_allclose(
        df["future_vol_5"].iloc[0],
        expected,
    )

def test_future_return_targets():

    df = pd.DataFrame({
        "log_return": [
            0.01,
            0.02,
            -0.01,
            0.03,
            0.01,
            0.02,
        ]
    })

    df = add_future_return_targets(df)

    expected = (
        0.02
        - 0.01
        + 0.03
        + 0.01
        + 0.02
    )

    np.testing.assert_allclose(
        df["future_return_5"].iloc[0],
        expected,
    )