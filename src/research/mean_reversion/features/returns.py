from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# MEAN REVERSION — RETURN FEATURES
# ============================================================
#
# PURPOSE
# -------
# This module creates historical return / price-displacement
# features that will later be used to investigate whether
# unusually large movements are followed by mean reversion.
#
# IMPORTANT
# ---------
# These are FEATURES, not trading signals.
#
# Nothing in this file decides:
#   - when to enter
#   - when to exit
#   - stop loss
#   - take profit
#   - position size
#
# It only describes what the market has already done.
#
# NO FUTURE INFORMATION
# ---------------------
# Every feature is calculated using the current observation
# and/or observations that occurred before it.
#
# Future returns will NOT be calculated here.
# Forward-looking variables belong to the research/target layer.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# Horizons are expressed in 1-minute bars because our dataset
# is NQ/MNQ 1-minute data.
#
# Example:
#
# return_5  -> movement over the previous 5 one-minute bars
# return_30 -> movement over the previous 30 one-minute bars
#
RETURN_WINDOWS = (1, 5, 10, 15, 30, 60)


# ============================================================
# SIMPLE RETURNS
# ============================================================

def add_simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical close-to-close returns.

    For a window N:

        return_N = close_t / close_(t-N) - 1

    This answers:

        "How much has price moved over the previous N bars?"

    Example:

        close_t       = 20,100
        close_(t-5)   = 20,000

        return_5 = 20,100 / 20,000 - 1
                 = 0.005
                 = +0.5%

    The calculation only looks backward, so it is safe as a
    feature for later research.
    """

    df = df.copy()

    for window in RETURN_WINDOWS:
        df[f"return_{window}"] = (
            df["close"] / df["close"].shift(window)
        ) - 1.0

    return df


# ============================================================
# LOG RETURNS
# ============================================================

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical logarithmic returns.

    For a one-bar return:

        log_return_1 = log(close_t / close_(t-1))

    For an N-bar return:

        log_return_N = log(close_t / close_(t-N))

    Log returns are useful in quantitative research because
    they are additive across consecutive periods.

    Example:

        log_return_5

    represents the logarithmic price movement over the
    previous 5 bars.
    """

    df = df.copy()

    for window in RETURN_WINDOWS:
        df[f"log_return_{window}"] = np.log(
            df["close"] / df["close"].shift(window)
        )

    return df


# ============================================================
# PRICE DISPLACEMENT
# ============================================================

def add_price_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical price displacement in NQ points.

    For a window N:

        displacement_N = close_t - close_(t-N)

    Unlike percentage returns, this keeps the movement in
    actual NQ points.

    Example:

        current close      = 20,100
        close 15 bars ago  = 20,060

        displacement_15 = +40 points

    This will be useful later when comparing the magnitude
    of price movements with volatility and VWAP distance.
    """

    df = df.copy()

    for window in RETURN_WINDOWS:
        df[f"displacement_{window}"] = (
            df["close"] - df["close"].shift(window)
        )

    return df


# ============================================================
# DIRECTION
# ============================================================

def add_return_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the direction of the historical movement.

    Values:

        +1  -> price increased over the window
         0  -> price unchanged
        -1  -> price decreased over the window

    This is deliberately kept separate from the magnitude of
    the movement.

    Example:

        displacement_15 = +50
        direction_15   = +1

    or:

        displacement_15 = -50
        direction_15   = -1
    """

    df = df.copy()

    for window in RETURN_WINDOWS:
        displacement = df[f"displacement_{window}"]

        df[f"direction_{window}"] = np.sign(displacement)

    return df


# ============================================================
# MASTER FUNCTION
# ============================================================

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete return feature set.

    The modules are intentionally separated so that each
    transformation can be tested independently.

    Pipeline:

        raw OHLC
           |
           +--> simple returns
           |
           +--> log returns
           |
           +--> price displacement
           |
           +--> movement direction
           |
           v
        return feature set

    No strategy logic is included here.
    """

    required_columns = {"close"}

    missing = required_columns - set(df.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = add_simple_returns(df)
    df = add_log_returns(df)
    df = add_price_displacement(df)
    df = add_return_direction(df)

    return df
