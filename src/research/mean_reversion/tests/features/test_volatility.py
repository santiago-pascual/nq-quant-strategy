from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.mean_reversion.features.volatility import (
    add_realized_volatility,
    add_true_range,
    add_rolling_range,
    add_average_true_range,
    add_volatility_ratios,
    add_normalized_range,
    add_volatility_features,
)


# ============================================================
# TEST DATA
# ============================================================


@pytest.fixture
def ohlc_data() -> pd.DataFrame:
    """
    Deterministic OHLC dataset used to verify the volatility
    calculations.
    """

    return pd.DataFrame(
        {
            "open": [
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
            ],
            "high": [
                101.0,
                102.0,
                103.0,
                105.0,
                104.0,
                107.0,
                111.0,
                109.0,
                113.0,
                116.0,
            ],
            "low": [
                99.0,
                100.0,
                101.0,
                103.0,
                102.0,
                105.0,
                109.0,
                107.0,
                111.0,
                114.0,
            ],
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
            ],
        }
    )


# ============================================================
# REALIZED VOLATILITY
# ============================================================


def test_realized_volatility_requires_log_returns(
    ohlc_data,
):

    with pytest.raises(KeyError):
        add_realized_volatility(ohlc_data)


def test_realized_volatility_5(
    ohlc_data,
):

    df = ohlc_data.copy()

    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    result = add_realized_volatility(df)

    returns = df["log_return_1"]

    expected = np.sqrt((returns.iloc[1:6] ** 2).sum())

    assert np.isclose(
        result.loc[5, "realized_vol_5"],
        expected,
    )


# ============================================================
# TRUE RANGE
# ============================================================


def test_true_range_first_bar(
    ohlc_data,
):

    result = add_true_range(ohlc_data)

    assert result.loc[0, "true_range"] == 2.0


def test_true_range_with_previous_close(
    ohlc_data,
):

    result = add_true_range(ohlc_data)

    expected = max(
        107.0 - 105.0,
        abs(107.0 - 103.0),
        abs(105.0 - 103.0),
    )

    assert result.loc[5, "true_range"] == expected


# ============================================================
# ROLLING RANGE
# ============================================================


def test_rolling_range_5(
    ohlc_data,
):

    result = add_rolling_range(ohlc_data)

    expected = ohlc_data["high"].iloc[1:6].max() - ohlc_data["low"].iloc[1:6].min()

    assert result.loc[5, "rolling_range_5"] == expected


# ============================================================
# ATR
# ============================================================


def test_atr_5(
    ohlc_data,
):

    df = add_true_range(ohlc_data)

    result = add_average_true_range(df)

    expected = df["true_range"].iloc[1:6].mean()

    assert np.isclose(
        result.loc[5, "atr_5"],
        expected,
    )


# ============================================================
# VOLATILITY RATIOS
# ============================================================


def test_volatility_ratios():

    closes = np.arange(
        100.0,
        151.0,
        1.0,
    )

    df = pd.DataFrame(
        {
            "close": closes,
        }
    )

    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    df = add_realized_volatility(df)

    result = add_volatility_ratios(df)

    expected = result.loc[50, "realized_vol_5"] / result.loc[50, "realized_vol_30"]

    assert np.isfinite(expected)

    assert np.isclose(
        result.loc[50, "vol_ratio_5_30"],
        expected,
    )


# ============================================================
# NORMALIZED RANGE
# ============================================================


def test_normalized_range(
    ohlc_data,
):

    df = ohlc_data.copy()

    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    df = add_realized_volatility(df)
    df = add_true_range(df)
    df = add_rolling_range(df)
    df = add_average_true_range(df)

    result = add_normalized_range(df)

    expected = result.loc[5, "rolling_range_5"] / result.loc[5, "atr_5"]

    assert np.isclose(
        result.loc[5, "normalized_range_5"],
        expected,
    )


# ============================================================
# MASTER FUNCTION
# ============================================================


def test_master_function_creates_all_features(
    ohlc_data,
):

    result = add_volatility_features(ohlc_data)

    for window in (5, 15, 30, 60):
        assert f"realized_vol_{window}" in result.columns
        assert f"rolling_range_{window}" in result.columns
        assert f"atr_{window}" in result.columns
        assert f"normalized_range_{window}" in result.columns

    assert "true_range" in result.columns
    assert "vol_ratio_5_30" in result.columns


# ============================================================
# ROW COUNT
# ============================================================


def test_volatility_features_preserve_row_count(
    ohlc_data,
):

    result = add_volatility_features(ohlc_data)

    assert len(result) == len(ohlc_data)


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_missing_ohlc_columns_raise_error():

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(KeyError):
        add_volatility_features(df)


# ============================================================
# NO LOOK-AHEAD
# ============================================================


def test_volatility_features_do_not_depend_on_future_data():

    df_original = pd.DataFrame(
        {
            "high": [
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
            ],
            "low": [
                99.0,
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
        }
    )

    df_modified = df_original.copy()

    # Change ONLY the future observation.
    df_modified.loc[5, "high"] = 1000.0
    df_modified.loc[5, "low"] = 900.0
    df_modified.loc[5, "close"] = 950.0

    original = add_volatility_features(df_original)

    modified = add_volatility_features(df_modified)

    # Observation 4 must not change.
    for column in [
        "realized_vol_5",
        "rolling_range_5",
        "atr_5",
        "normalized_range_5",
    ]:
        assert np.isclose(
            original.loc[4, column],
            modified.loc[4, column],
            equal_nan=True,
        )
