import numpy as np
import pandas as pd

from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)


def test_return_features():

    df = pd.DataFrame({"close": [100.0, 101.0, 103.0]})

    df = add_return_features(df)

    expected_returns = [
        np.nan,
        0.01,
        103 / 101 - 1,
    ]

    expected_log_returns = [
        np.nan,
        np.log(101 / 100),
        np.log(103 / 101),
    ]

    assert np.isnan(df["return"].iloc[0])
    assert np.isnan(df["log_return"].iloc[0])

    np.testing.assert_allclose(df["return"].iloc[1:], expected_returns[1:])

    np.testing.assert_allclose(df["log_return"].iloc[1:], expected_log_returns[1:])


def test_volatility_features():

    log_returns = [0.01] * 60

    df = pd.DataFrame({
        "log_return": log_returns
    })

    df = add_volatility_features(df)

    expected_vol_5 = np.sqrt(5 * 0.01**2)
    expected_vol_30 = np.sqrt(30 * 0.01**2)
    expected_vol_60 = np.sqrt(60 * 0.01**2)

    expected_ratio_5_30 = (
        expected_vol_5 / expected_vol_30
    )

    expected_ratio_5_60 = (
        expected_vol_5 / expected_vol_60
    )

    assert np.isnan(df["realized_vol_5"].iloc[3])

    np.testing.assert_allclose(
        df["realized_vol_5"].iloc[4],
        expected_vol_5,
    )

    np.testing.assert_allclose(
        df["realized_vol_30"].iloc[29],
        expected_vol_30,
    )

    np.testing.assert_allclose(
        df["realized_vol_60"].iloc[59],
        expected_vol_60,
    )

    np.testing.assert_allclose(
        df["vol_ratio_5_30"].iloc[59],
        expected_ratio_5_30,
    )

    np.testing.assert_allclose(
        df["vol_ratio_5_60"].iloc[59],
        expected_ratio_5_60,
    )