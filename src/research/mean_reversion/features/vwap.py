from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# MEAN REVERSION — VWAP FEATURES
# ============================================================
#
# PURPOSE
# -------
# VWAP (Volume Weighted Average Price) is a price reference
# weighted by traded volume.
#
# In Mean Reversion research, we are interested in questions
# such as:
#
#   - Does price tend to return toward VWAP?
#   - Does the probability of reversion increase as distance
#     from VWAP increases?
#   - Does the relationship depend on volatility?
#   - Does it behave differently during different sessions?
#
# IMPORTANT
# ---------
# This module creates FEATURES only.
#
# It does NOT define:
#
#   - entry rules
#   - exits
#   - stop losses
#   - targets
#   - position sizing
#
# ============================================================
#
# VWAP DEFINITION
# ---------------
#
# For each session:
#
#             sum(price_i * volume_i)
# VWAP_t =    ---------------------
#                 sum(volume_i)
#
# We use the typical price:
#
#             high + low + close
# TP =        ---------------
#                    3
#
# VWAP is calculated cumulatively from the beginning of each
# session up to the current observation.
#
# Therefore the VWAP at time t never uses future observations.
#
# ============================================================


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "high",
    "low",
    "close",
    "volume",
}


# ============================================================
# SESSION IDENTIFICATION
# ============================================================


def add_vwap_session_id(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the session identifier used for VWAP accumulation.

    Priority:

    1. Existing `session_date` column.
    2. Datetime index.
    3. `timestamp ET` column.

    The objective is to ensure that cumulative VWAP does not
    accidentally continue from one trading session into another.

    If `session_date` already exists, we preserve its meaning
    rather than reconstructing it.
    """

    df = df.copy()

    if "session_date" in df.columns:
        df["_vwap_session"] = df["session_date"].astype(str)

        return df

    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index

        if index.tz is not None:
            index = index.tz_convert("America/New_York")

        df["_vwap_session"] = index.date.astype(str)

        return df

    if "timestamp ET" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp ET"],
            errors="coerce",
        )

        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("America/New_York")

        else:
            timestamps = timestamps.dt.tz_convert("America/New_York")

        df["_vwap_session"] = timestamps.dt.date.astype(str)

        return df

    raise KeyError(
        "Cannot determine VWAP session. "
        "Provide 'session_date', a DatetimeIndex, "
        "or 'timestamp ET'."
    )


# ============================================================
# TYPICAL PRICE
# ============================================================


def add_typical_price(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the typical price.

        typical_price =
            (high + low + close) / 3

    This is the price representation used by our VWAP.
    """

    missing = {"high", "low", "close"} - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

    return df


# ============================================================
# SESSION VWAP
# ============================================================


def add_session_vwap(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate cumulative session VWAP.

    For every observation:

        cumulative_pv =
            sum(typical_price * volume)

        cumulative_volume =
            sum(volume)

        VWAP =
            cumulative_pv / cumulative_volume

    Both cumulative sums are reset at the beginning of each
    session.

    IMPORTANT:

    The current observation is included.

    This means VWAP_t represents the information available
    after processing bar t.

    No future bar contributes to VWAP_t.
    """

    required = {
        "typical_price",
        "volume",
        "_vwap_session",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    price_volume = df["typical_price"] * df["volume"]

    cumulative_pv = price_volume.groupby(
        df["_vwap_session"],
        sort=False,
    ).cumsum()

    cumulative_volume = (
        df["volume"]
        .groupby(
            df["_vwap_session"],
            sort=False,
        )
        .cumsum()
    )

    df["vwap"] = cumulative_pv / cumulative_volume.replace(0, np.nan)

    return df


# ============================================================
# VWAP DISTANCE
# ============================================================


def add_vwap_distance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the distance between the current close and VWAP.

        vwap_distance =
            close - vwap

    Positive value:
        price is above VWAP.

    Negative value:
        price is below VWAP.

    This is one of the core Mean Reversion variables because
    it tells us how far price has moved away from a potential
    intraday equilibrium reference.
    """

    required = {
        "close",
        "vwap",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["vwap_distance"] = df["close"] - df["vwap"]

    return df


# ============================================================
# VWAP PERCENTAGE DISTANCE
# ============================================================


def add_vwap_percentage_distance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate percentage distance from VWAP.

        vwap_distance_pct =
            (close - vwap) / vwap

    This makes the distance comparable across different NQ
    price levels.
    """

    required = {
        "close",
        "vwap",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["vwap_distance_pct"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    return df


# ============================================================
# VOLATILITY-NORMALIZED VWAP DISTANCE
# ============================================================


def add_normalized_vwap_distance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize VWAP distance by a volatility scale.

    Preferred scale:

        atr_30

    Formula:

        normalized_vwap_distance =
            (close - vwap) / atr_30

    This allows us to distinguish:

        "price is 30 points from VWAP"

    from:

        "price is 30 points from VWAP relative to the
         volatility currently present in the market."

    If ATR_30 is unavailable, this function raises an error
    rather than silently substituting another measure.

    The choice of ATR_30 is a RESEARCH DEFINITION, not a
    strategy parameter.
    """

    required = {
        "close",
        "vwap",
        "atr_30",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["normalized_vwap_distance"] = (df["close"] - df["vwap"]) / df["atr_30"].replace(
        0, np.nan
    )

    return df


# ============================================================
# VWAP SIDE
# ============================================================


def add_vwap_side(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify whether price is above, below, or exactly at VWAP.

        +1 -> above VWAP
         0 -> exactly at VWAP
        -1 -> below VWAP
    """

    if "vwap_distance" not in df.columns:
        raise KeyError(
            "Missing required column: 'vwap_distance'. Run add_vwap_distance() first."
        )

    df = df.copy()

    df["vwap_side"] = np.sign(df["vwap_distance"])

    return df


# ============================================================
# MASTER FUNCTION
# ============================================================


def add_vwap_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete VWAP feature set.

    Pipeline:

        OHLCV
          |
          +--> session identification
          |
          +--> typical price
          |
          +--> cumulative session VWAP
          |
          +--> VWAP distance
          |
          +--> percentage distance
          |
          +--> volatility-normalized distance
          |
          +--> VWAP side
          |
          v
       VWAP FEATURES

    The function expects `atr_30` to already exist for the
    normalized distance feature.

    This keeps feature modules independent and explicit.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df = add_vwap_session_id(df)

    df = add_typical_price(df)

    df = add_session_vwap(df)

    df = add_vwap_distance(df)

    df = add_vwap_percentage_distance(df)

    df = add_normalized_vwap_distance(df)

    df = add_vwap_side(df)

    return df
