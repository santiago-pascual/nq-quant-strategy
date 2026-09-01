from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.mean_reversion.features.returns import (
    add_simple_returns,
    add_log_returns,
    add_price_displacement,
    add_return_direction,
    add_return_features,
)


# ============================================================
# TEST DATA
# ============================================================
#
# We use a deterministic artificial price series.
#
# This makes the expected results known exactly and allows us
# to test the mathematical implementation independently from
# the real NQ dataset.
#
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
# SIMPLE RETURNS
# ============================================================


def test_simple_return_1(price_data):
    """
    Verify the one-bar simple return.

    Formula:

        return_1 = close_t / close_(t-1) - 1
    """

    result = add_simple_returns(price_data)

    expected = (101.0 / 100.0) - 1.0

    assert np.isclose(
        result.loc[1, "return_1"],
        expected,
    )


def test_simple_return_5(price_data):
    """
    Verify that return_5 compares the current close with the
    close exactly five bars in the past.
    """

    result = add_simple_returns(price_data)

    expected = (106.0 / 100.0) - 1.0

    assert np.isclose(
        result.loc[5, "return_5"],
        expected,
    )


def test_simple_returns_have_correct_initial_nan_count(price_data):
    """
    A return over N bars cannot exist until N previous
    observations are available.

    Therefore return_N must contain exactly N initial NaNs.
    """

    result = add_simple_returns(price_data)

    for window in (1, 5):
        assert result[f"return_{window}"].iloc[:window].isna().all()


# ============================================================
# LOG RETURNS
# ============================================================


def test_log_return_1(price_data):
    """
    Verify the one-bar logarithmic return.
    """

    result = add_log_returns(price_data)

    expected = np.log(101.0 / 100.0)

    assert np.isclose(
        result.loc[1, "log_return_1"],
        expected,
    )


def test_log_return_5(price_data):
    """
    Verify the five-bar logarithmic return.
    """

    result = add_log_returns(price_data)

    expected = np.log(106.0 / 100.0)

    assert np.isclose(
        result.loc[5, "log_return_5"],
        expected,
    )


# ============================================================
# PRICE DISPLACEMENT
# ============================================================


def test_price_displacement_1(price_data):
    """
    Verify one-bar displacement in actual price points.
    """

    result = add_price_displacement(price_data)

    assert result.loc[1, "displacement_1"] == 1.0


def test_price_displacement_5(price_data):
    """
    Verify five-bar displacement.
    """

    result = add_price_displacement(price_data)

    assert result.loc[5, "displacement_5"] == 6.0


# ============================================================
# DIRECTION
# ============================================================


def test_return_direction(price_data):
    """
    Verify directional classification:

        positive movement -> +1
        negative movement -> -1
        zero movement     ->  0
    """

    result = add_return_direction(add_price_displacement(price_data))

    assert result.loc[1, "direction_1"] == 1
    assert result.loc[4, "direction_1"] == -1
    assert result.loc[1, "direction_1"] in (-1, 0, 1)


def test_zero_movement_direction():
    """
    A zero price movement must produce direction = 0.
    """

    df = pd.DataFrame(
        {
            "close": [
                100.0,
                100.0,
            ]
        }
    )

    result = add_price_displacement(df)
    result = add_return_direction(result)

    assert result.loc[1, "direction_1"] == 0


# ============================================================
# MASTER FUNCTION
# ============================================================


def test_master_function_creates_all_features(price_data):
    """
    Verify that add_return_features() creates the complete
    expected feature set.
    """

    result = add_return_features(price_data)

    for window in (1, 5, 10, 15, 30, 60):
        assert f"return_{window}" in result.columns
        assert f"log_return_{window}" in result.columns
        assert f"displacement_{window}" in result.columns
        assert f"direction_{window}" in result.columns


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_missing_close_column_raises_error():
    """
    The feature engine requires a close column.
    """

    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
        }
    )

    with pytest.raises(KeyError):
        add_return_features(df)


# ============================================================
# NO DATA LOSS
# ============================================================


def test_feature_engine_preserves_row_count(price_data):
    """
    Feature generation must never remove observations.

    NaNs are expected at the beginning of rolling/lagged
    calculations, but the rows themselves must remain.
    """

    result = add_return_features(price_data)

    assert len(result) == len(price_data)


# ============================================================
# NO FUTURE DATA
# ============================================================


def test_return_features_do_not_depend_on_future_prices():
    """
    Critical anti-lookahead test.

    We calculate the features for the same observation twice.

    Then we change ONLY a future price.

    The historical feature value must remain unchanged.

    If it changes, the feature is accidentally using future
    information.
    """

    df_original = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                104.0,
                103.0,
                106.0,
            ]
        }
    )

    df_modified = df_original.copy()

    # Change ONLY the future observation.
    df_modified.loc[5, "close"] = 1000.0

    original = add_return_features(df_original)
    modified = add_return_features(df_modified)

    # Observation 4 must not be affected by changing
    # observation 5.
    for column in [
        "return_1",
        "return_5",
        "log_return_1",
        "log_return_5",
        "displacement_1",
        "displacement_5",
        "direction_1",
        "direction_5",
    ]:
        assert np.isclose(
            original.loc[4, column],
            modified.loc[4, column],
            equal_nan=True,
        )
