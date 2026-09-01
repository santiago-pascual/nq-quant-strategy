from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# MEAN REVERSION — VOLATILITY FEATURES
# ============================================================
#
# PURPOSE
# -------
# This module measures the volatility and recent price range
# of the market.
#
# These are FEATURES, not strategy rules.
#
# We are NOT deciding here:
#   - when to enter
#   - when to exit
#   - what volatility regime is "good"
#   - what parameter should be optimized
#
# The purpose is to describe the market state so that the
# research layer can later investigate whether mean reversion
# behaves differently under different volatility conditions.
#
# IMPORTANT
# ---------
# All calculations use information available at the current
# observation or earlier.
#
# No future data is used.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# All windows are measured in 1-minute bars.
#
# 5  -> approximately 5 minutes
# 15 -> approximately 15 minutes
# 30 -> approximately 30 minutes
# 60 -> approximately 60 minutes
#
VOLATILITY_WINDOWS = (5, 15, 30, 60)


# ============================================================
# REALIZED VOLATILITY
# ============================================================


def add_realized_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling realized volatility from historical
    log returns.

    For a window N:

        realized_vol_N
            = sqrt(
                sum(log_return_1^2 over previous N bars)
              )

    This measures the accumulated magnitude of recent
    price movement.

    IMPORTANT:

    This is NOT a forecast of future volatility.

    It describes volatility that has already occurred.

    Example:

        realized_vol_30

    answers approximately:

        "How much movement has occurred during the last
         30 one-minute bars?"

    The function expects `log_return_1` to already exist.
    """

    if "log_return_1" not in df.columns:
        raise KeyError(
            "Missing required column: 'log_return_1'. Run add_log_returns() first."
        )

    df = df.copy()

    squared_returns = df["log_return_1"] ** 2

    for window in VOLATILITY_WINDOWS:
        df[f"realized_vol_{window}"] = np.sqrt(squared_returns.rolling(window).sum())

    return df


# ============================================================
# ROLLING TRUE RANGE
# ============================================================


def add_true_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the standard True Range for every bar.

    For each observation:

        TR = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )

    True Range captures both:

        1. movement inside the current bar
        2. gaps relative to the previous close

    This is useful because close-to-close returns alone do not
    describe the full intrabar price range.
    """

    required = {"high", "low", "close"}

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    previous_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]

    high_previous_close = (df["high"] - previous_close).abs()

    low_previous_close = (df["low"] - previous_close).abs()

    df["true_range"] = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close,
        ],
        axis=1,
    ).max(axis=1)

    return df


# ============================================================
# ROLLING RANGE
# ============================================================


def add_rolling_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the total high-low range over rolling windows.

    For a window N:

        rolling_range_N =
            rolling maximum(high)
            -
            rolling minimum(low)

    This measures the total price territory covered during
    the recent window.

    Unlike realized volatility, this is expressed directly
    in NQ price points.
    """

    required = {"high", "low"}

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    for window in VOLATILITY_WINDOWS:
        rolling_high = df["high"].rolling(window).max()
        rolling_low = df["low"].rolling(window).min()

        df[f"rolling_range_{window}"] = rolling_high - rolling_low

    return df


# ============================================================
# AVERAGE TRUE RANGE
# ============================================================


def add_average_true_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling Average True Range (ATR-like feature).

    For a window N:

        ATR_N = mean(True Range over N bars)

    This gives us a volatility scale in actual NQ points.

    We use it as a RESEARCH FEATURE.

    We are not assuming that ATR should be used as a stop,
    target, or entry condition.
    """

    if "true_range" not in df.columns:
        raise KeyError(
            "Missing required column: 'true_range'. Run add_true_range() first."
        )

    df = df.copy()

    for window in VOLATILITY_WINDOWS:
        df[f"atr_{window}"] = df["true_range"].rolling(window).mean()

    return df


# ============================================================
# VOLATILITY RATIOS
# ============================================================


def add_volatility_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare short-term volatility with longer-term volatility.

    These ratios help describe whether volatility is currently
    expanding or contracting relative to its recent baseline.

    Example:

        vol_ratio_5_30 =
            realized_vol_5 / realized_vol_30

    A high value means recent 5-bar volatility is large
    relative to the recent 30-bar volatility scale.

    IMPORTANT:

    No threshold is defined here.

    The research stage will determine whether these ratios
    contain useful information.
    """

    required = {
        "realized_vol_5",
        "realized_vol_15",
        "realized_vol_30",
        "realized_vol_60",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["vol_ratio_5_15"] = df["realized_vol_5"] / df["realized_vol_15"]

    df["vol_ratio_5_30"] = df["realized_vol_5"] / df["realized_vol_30"]

    df["vol_ratio_5_60"] = df["realized_vol_5"] / df["realized_vol_60"]

    df["vol_ratio_15_30"] = df["realized_vol_15"] / df["realized_vol_30"]

    df["vol_ratio_15_60"] = df["realized_vol_15"] / df["realized_vol_60"]

    df["vol_ratio_30_60"] = df["realized_vol_30"] / df["realized_vol_60"]

    return df


# ============================================================
# NORMALIZED RANGE
# ============================================================


def add_normalized_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize recent price range by a volatility scale.

    Example:

        normalized_range_30 =
            rolling_range_30 / atr_30

    This allows us to distinguish between:

        "price moved 50 points"

    and:

        "price moved 50 points relative to the volatility
         that normally exists in this environment."

    This distinction is important for Mean Reversion research.
    """

    required = {
        "rolling_range_5",
        "rolling_range_15",
        "rolling_range_30",
        "rolling_range_60",
        "atr_5",
        "atr_15",
        "atr_30",
        "atr_60",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    for window in VOLATILITY_WINDOWS:
        df[f"normalized_range_{window}"] = (
            df[f"rolling_range_{window}"] / df[f"atr_{window}"]
        )

    return df


# ============================================================
# MASTER FUNCTION
# ============================================================


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete volatility feature set.

    Pipeline:

        OHLC
         |
         +--> True Range
         |
         +--> ATR
         |
         +--> Rolling Range
         |
         +--> Log Returns
         |       |
         |       +--> Realized Volatility
         |
         +--> Volatility Ratios
         |
         +--> Normalized Range
         |
         v
      VOLATILITY FEATURES

    The function automatically creates the required
    log-return feature if it is not already present.

    No strategy logic is included.
    """

    required = {
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    # Realized volatility needs one-bar log returns.
    if "log_return_1" not in df.columns:
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    df = add_realized_volatility(df)

    df = add_true_range(df)

    df = add_rolling_range(df)

    df = add_average_true_range(df)

    df = add_volatility_ratios(df)

    df = add_normalized_range(df)

    return df
