from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.mean_reversion.features.zscore import (
    add_rolling_means,
    add_rolling_std,
    add_mean_distance,
    add_price_zscores,
    add_log_price_zscores,
    add_absolute_zscores,
    add_zscore_direction,
    add_zscore_features,
)


# ============================================================
# TEST DATA
# ============================================================


@pytest.fixture
def price_data() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                104.0,
                103.0,
                106.0,
                110.0,
                108.0,
                112.0,
                115.0,
            ]
        }
    )


# ============================================================
# ROLLING MEAN
# ============================================================


def test_rolling_mean_5(
    price_data,
):

    result = add_rolling_means(price_data)

    expected = np.mean(
        [
            101.0,
            102.0,
            104.0,
            103.0,
            106.0,
        ]
    )

    assert np.isclose(
        result.loc[5, "rolling_mean_5"],
        expected,
    )


# ============================================================
# ROLLING STANDARD DEVIATION
# ============================================================


def test_rolling_std_5(
    price_data,
):

    result = add_rolling_std(price_data)

    expected = np.std(
        [
            101.0,
            102.0,
            104.0,
            103.0,
            106.0,
        ],
        ddof=1,
    )

    assert np.isclose(
        result.loc[5, "rolling_std_5"],
        expected,
    )


# ============================================================
# MEAN DISTANCE
# ============================================================


def test_mean_distance_5(
    price_data,
):

    df = add_rolling_means(price_data)

    result = add_mean_distance(df)

    expected = 106.0 - np.mean(
        [
            101.0,
            102.0,
            104.0,
            103.0,
            106.0,
        ]
    )

    assert np.isclose(
        result.loc[5, "mean_distance_5"],
        expected,
    )


# ============================================================
# PRICE Z-SCORE
# ============================================================


def test_price_zscore_5(
    price_data,
):

    df = add_rolling_means(price_data)

    df = add_rolling_std(df)

    result = add_price_zscores(df)

    values = np.array(
        [
            101.0,
            102.0,
            104.0,
            103.0,
            106.0,
        ]
    )

    expected = (106.0 - values.mean()) / values.std(ddof=1)

    assert np.isclose(
        result.loc[5, "zscore_5"],
        expected,
    )


# ============================================================
# LOG PRICE Z-SCORE
# ============================================================


def test_log_price_zscore_5(
    price_data,
):

    result = add_log_price_zscores(price_data)

    values = np.log(
        np.array(
            [
                101.0,
                102.0,
                104.0,
                103.0,
                106.0,
            ]
        )
    )

    expected = (np.log(106.0) - values.mean()) / values.std(ddof=1)

    assert np.isclose(
        result.loc[5, "log_zscore_5"],
        expected,
    )


# ============================================================
# ABSOLUTE Z-SCORE
# ============================================================


def test_absolute_zscore(
    price_data,
):

    df = add_zscore_features(price_data)

    expected = abs(df.loc[5, "zscore_5"])

    assert np.isclose(
        df.loc[5, "abs_zscore_5"],
        expected,
    )


# ============================================================
# Z-SCORE DIRECTION
# ============================================================


def test_zscore_direction(
    price_data,
):

    df = add_zscore_features(price_data)

    for window in (
        5,
        15,
        30,
        60,
    ):
        column = f"zscore_direction_{window}"

        assert column in df.columns

    assert df.loc[5, "zscore_direction_5"] == 1


# ============================================================
# ZERO STANDARD DEVIATION
# ============================================================


def test_zero_standard_deviation_returns_nan():

    df = pd.DataFrame(
        {
            "close": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ]
        }
    )

    result = add_zscore_features(df)

    assert pd.isna(result.loc[4, "zscore_5"])


# ============================================================
# MASTER FUNCTION
# ============================================================


def test_master_function_creates_all_features(
    price_data,
):

    result = add_zscore_features(price_data)

    for window in (
        5,
        15,
        30,
        60,
    ):
        assert f"rolling_mean_{window}" in result.columns

        assert f"rolling_std_{window}" in result.columns

        assert f"mean_distance_{window}" in result.columns

        assert f"zscore_{window}" in result.columns

        assert f"abs_zscore_{window}" in result.columns

        assert f"zscore_direction_{window}" in result.columns

        assert f"log_zscore_{window}" in result.columns


# ============================================================
# ROW COUNT
# ============================================================


def test_zscore_features_preserve_row_count(
    price_data,
):

    result = add_zscore_features(price_data)

    assert len(result) == len(price_data)


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_missing_close_raises_error():

    df = pd.DataFrame(
        {
            "open": [
                100.0,
                101.0,
            ]
        }
    )

    with pytest.raises(KeyError):
        add_zscore_features(df)


# ============================================================
# NO LOOK-AHEAD
# ============================================================


def test_zscore_features_do_not_depend_on_future_data():

    df_original = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ]
        }
    )

    df_modified = df_original.copy()

    # Change ONLY the future observation.
    df_modified.loc[5, "close"] = 1000.0

    original = add_zscore_features(df_original)

    modified = add_zscore_features(df_modified)

    # Observation 4 must be unchanged.
    for column in [
        "rolling_mean_5",
        "rolling_std_5",
        "mean_distance_5",
        "zscore_5",
        "abs_zscore_5",
        "zscore_direction_5",
        "log_zscore_5",
    ]:
        assert np.isclose(
            original.loc[4, column],
            modified.loc[4, column],
            equal_nan=True,
        )
