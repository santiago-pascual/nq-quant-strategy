from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# ============================================================
# TEST DATA
# ============================================================


@pytest.fixture
def complete_ohlcv_data() -> pd.DataFrame:

    n = 200

    close = 100.0 + np.sin(np.arange(n) / 5.0) + np.arange(n) * 0.05

    return pd.DataFrame(
        {
            "timestamp ET": pd.date_range(
                "2026-01-05 09:30",
                periods=n,
                freq="min",
                tz="America/New_York",
            ),
            "session_date": [
                "2026-01-05" if i < 100 else "2026-01-06" for i in range(n)
            ],
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(
                n,
                1000.0,
            ),
        }
    )


# ============================================================
# MASTER PIPELINE
# ============================================================


def test_feature_engine_runs(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    assert isinstance(
        result,
        pd.DataFrame,
    )


# ============================================================
# ROW COUNT
# ============================================================


def test_feature_engine_preserves_row_count(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    assert len(result) == len(complete_ohlcv_data)


# ============================================================
# INDEX
# ============================================================


def test_feature_engine_preserves_index(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    assert result.index.equals(complete_ohlcv_data.index)


# ============================================================
# RETURNS
# ============================================================


def test_returns_features_are_present(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    assert "return_1" in result.columns
    assert "return_5" in result.columns
    assert "log_return_1" in result.columns


# ============================================================
# VOLATILITY
# ============================================================


def test_volatility_features_are_present(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    assert "realized_vol_5" in result.columns
    assert "realized_vol_30" in result.columns
    assert "atr_30" in result.columns


# ============================================================
# VWAP
# ============================================================


def test_vwap_features_are_present(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    assert "vwap" in result.columns
    assert "vwap_distance" in result.columns
    assert "vwap_distance_pct" in result.columns
    assert "normalized_vwap_distance" in result.columns


# ============================================================
# Z-SCORE
# ============================================================


def test_zscore_features_are_present(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    for window in (
        5,
        15,
        30,
        60,
    ):
        assert f"zscore_{window}" in result.columns

        assert f"rolling_mean_{window}" in result.columns


# ============================================================
# OU
# ============================================================


def test_ou_features_are_present(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    for window in (
        30,
        60,
        120,
    ):
        assert f"autocorrelation_{window}" in result.columns

        assert f"ar1_coefficient_{window}" in result.columns

        assert f"half_life_{window}" in result.columns


# ============================================================
# ORIGINAL DATA PRESERVED
# ============================================================


def test_original_ohlcv_columns_are_preserved(
    complete_ohlcv_data,
):

    result = build_mean_reversion_features(complete_ohlcv_data)

    for column in [
        "timestamp ET",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        assert column in result.columns


# ============================================================
# NO LOOK-AHEAD — INTEGRATION LEVEL
# ============================================================


def test_feature_engine_has_no_future_dependency(
    complete_ohlcv_data,
):

    original = build_mean_reversion_features(complete_ohlcv_data)

    modified_data = complete_ohlcv_data.copy()

    # Modify ONLY the final future observation.
    modified_data.loc[
        199,
        "high",
    ] = 10000.0

    modified_data.loc[
        199,
        "low",
    ] = 9000.0

    modified_data.loc[
        199,
        "close",
    ] = 9500.0

    modified_data.loc[
        199,
        "volume",
    ] = 9999999.0

    modified = build_mean_reversion_features(modified_data)

    # A sufficiently earlier observation must not change.
    checkpoint = 150

    for column in [
        "vwap",
        "zscore_30",
        "zscore_60",
        "half_life_30",
        "half_life_60",
    ]:
        assert np.isclose(
            original.loc[
                checkpoint,
                column,
            ],
            modified.loc[
                checkpoint,
                column,
            ],
            equal_nan=True,
        )


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_non_dataframe_raises_error():

    with pytest.raises(TypeError):
        build_mean_reversion_features([100, 101, 102])
