from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.mean_reversion.features.vwap import (
    add_typical_price,
    add_session_vwap,
    add_vwap_distance,
    add_vwap_percentage_distance,
    add_normalized_vwap_distance,
    add_vwap_side,
    add_vwap_features,
)


# ============================================================
# TEST DATA
# ============================================================


@pytest.fixture
def vwap_data() -> pd.DataFrame:
    """
    Deterministic two-session dataset.

    The second session is deliberately different so we can
    verify that VWAP resets correctly between sessions.
    """

    return pd.DataFrame(
        {
            "timestamp ET": pd.to_datetime(
                [
                    "2026-01-05 09:30",
                    "2026-01-05 09:31",
                    "2026-01-05 09:32",
                    "2026-01-06 09:30",
                    "2026-01-06 09:31",
                    "2026-01-06 09:32",
                ]
            ).tz_localize("America/New_York"),
            "session_date": [
                "2026-01-05",
                "2026-01-05",
                "2026-01-05",
                "2026-01-06",
                "2026-01-06",
                "2026-01-06",
            ],
            "high": [
                101.0,
                103.0,
                105.0,
                201.0,
                203.0,
                205.0,
            ],
            "low": [
                99.0,
                101.0,
                103.0,
                199.0,
                201.0,
                203.0,
            ],
            "close": [
                100.0,
                102.0,
                104.0,
                200.0,
                202.0,
                204.0,
            ],
            "volume": [
                100.0,
                200.0,
                300.0,
                100.0,
                200.0,
                300.0,
            ],
            "atr_30": [
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
        }
    )


# ============================================================
# TYPICAL PRICE
# ============================================================


def test_typical_price(vwap_data):

    result = add_typical_price(vwap_data)

    expected = (101.0 + 99.0 + 100.0) / 3.0

    assert np.isclose(
        result.loc[0, "typical_price"],
        expected,
    )


# ============================================================
# SESSION VWAP
# ============================================================


def test_first_bar_vwap_equals_typical_price(
    vwap_data,
):

    df = add_typical_price(vwap_data)

    df = df.assign(_vwap_session=df["session_date"])

    result = add_session_vwap(df)

    assert np.isclose(
        result.loc[0, "vwap"],
        result.loc[0, "typical_price"],
    )


def test_session_vwap_weighted_average(
    vwap_data,
):

    df = add_typical_price(vwap_data)

    df = df.assign(_vwap_session=df["session_date"])

    result = add_session_vwap(df)

    prices = result.loc[
        0:2,
        "typical_price",
    ]

    volumes = result.loc[
        0:2,
        "volume",
    ]

    expected = (prices * volumes).sum() / volumes.sum()

    assert np.isclose(
        result.loc[2, "vwap"],
        expected,
    )


# ============================================================
# SESSION RESET
# ============================================================


def test_vwap_resets_at_new_session(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    # First bar of the second session must
    # use ONLY the second session's information.
    expected = (201.0 + 199.0 + 200.0) / 3.0

    assert np.isclose(
        result.loc[3, "vwap"],
        expected,
    )


# ============================================================
# DISTANCE
# ============================================================


def test_vwap_distance(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    expected = result.loc[2, "close"] - result.loc[2, "vwap"]

    assert np.isclose(
        result.loc[2, "vwap_distance"],
        expected,
    )


# ============================================================
# PERCENTAGE DISTANCE
# ============================================================


def test_vwap_percentage_distance(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    expected = (result.loc[2, "close"] - result.loc[2, "vwap"]) / result.loc[2, "vwap"]

    assert np.isclose(
        result.loc[2, "vwap_distance_pct"],
        expected,
    )


# ============================================================
# NORMALIZED DISTANCE
# ============================================================


def test_normalized_vwap_distance(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    expected = result.loc[2, "vwap_distance"] / result.loc[2, "atr_30"]

    assert np.isclose(
        result.loc[2, "normalized_vwap_distance"],
        expected,
    )


# ============================================================
# VWAP SIDE
# ============================================================


def test_vwap_side(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    assert result.loc[0, "vwap_side"] == 0
    assert result.loc[2, "vwap_side"] == 1
    assert result.loc[3, "vwap_side"] == 0


# ============================================================
# MASTER FUNCTION
# ============================================================


def test_master_function_creates_all_features(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    expected_columns = [
        "_vwap_session",
        "typical_price",
        "vwap",
        "vwap_distance",
        "vwap_distance_pct",
        "normalized_vwap_distance",
        "vwap_side",
    ]

    for column in expected_columns:
        assert column in result.columns


# ============================================================
# ROW COUNT
# ============================================================


def test_vwap_features_preserve_row_count(
    vwap_data,
):

    result = add_vwap_features(vwap_data)

    assert len(result) == len(vwap_data)


# ============================================================
# NO LOOK-AHEAD
# ============================================================


def test_vwap_does_not_use_future_data(
    vwap_data,
):

    original = add_vwap_features(vwap_data)

    modified_data = vwap_data.copy()

    # Change only a future observation.
    modified_data.loc[5, "high"] = 1000.0
    modified_data.loc[5, "low"] = 900.0
    modified_data.loc[5, "close"] = 950.0
    modified_data.loc[5, "volume"] = 999999.0

    modified = add_vwap_features(modified_data)

    # Observation 4 must remain identical.
    for column in [
        "vwap",
        "vwap_distance",
        "vwap_distance_pct",
        "normalized_vwap_distance",
        "vwap_side",
    ]:
        assert np.isclose(
            original.loc[4, column],
            modified.loc[4, column],
            equal_nan=True,
        )


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_missing_columns_raise_error():

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "volume": [100.0, 100.0],
        }
    )

    with pytest.raises(KeyError):
        add_vwap_features(df)
