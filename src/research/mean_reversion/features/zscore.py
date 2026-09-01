from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# MEAN REVERSION — Z-SCORE FEATURES
# ============================================================
#
# PURPOSE
# -------
# A Z-score measures how far the current price is from its
# recent mean in units of its recent standard deviation.
#
# Formula:
#
#                  price_t - mean_t
#     Z_t =        -----------------
#                    std_t
#
# Interpretation:
#
#     Z =  0   -> price is at its recent mean
#     Z = +1   -> price is one standard deviation above mean
#     Z = -1   -> price is one standard deviation below mean
#     Z = +2   -> price is two standard deviations above mean
#     Z = -2   -> price is two standard deviations below mean
#
# IMPORTANT
# ---------
# This module DOES NOT define entry thresholds.
#
# We are NOT assuming that:
#
#     Z < -2  -> buy
#     Z > +2  -> sell
#
# That will be investigated later using the actual NQ data.
#
# ============================================================
#
# NO LOOK-AHEAD
# ------------
#
# Rolling statistics use:
#
#     rolling(window).mean()
#     rolling(window).std()
#
# At time t these use observations up to t only.
#
# Future observations are never included.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# Windows are measured in 1-minute bars.
#
# These are research horizons, NOT optimized strategy
# parameters.
#
ZSCORE_WINDOWS = (5, 15, 30, 60)


# ============================================================
# ROLLING MEAN
# ============================================================


def add_rolling_means(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate rolling mean prices.

    For window N:

        rolling_mean_N(t)
            = mean(close[t-N+1 : t])

    The current close is included.

    No future observations are used.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    for window in ZSCORE_WINDOWS:
        df[f"rolling_mean_{window}"] = df["close"].rolling(window).mean()

    return df


# ============================================================
# ROLLING STANDARD DEVIATION
# ============================================================


def add_rolling_std(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate rolling standard deviation of price.

    Pandas uses sample standard deviation by default:

        ddof = 1

    This is explicitly retained so that the definition is
    deterministic and transparent.

    The standard deviation measures the dispersion of price
    around its rolling mean.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    for window in ZSCORE_WINDOWS:
        df[f"rolling_std_{window}"] = df["close"].rolling(window).std()

    return df


# ============================================================
# PRICE DISTANCE FROM MEAN
# ============================================================


def add_mean_distance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the distance between the current price and its
    rolling mean in actual NQ points.

        mean_distance_N =
            close - rolling_mean_N

    Positive:
        price is above its recent mean.

    Negative:
        price is below its recent mean.

    This feature is useful because Z-score normalizes this
    distance, while the raw distance preserves actual price
    displacement.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    for window in ZSCORE_WINDOWS:
        mean_column = f"rolling_mean_{window}"

        if mean_column not in df.columns:
            raise KeyError(
                f"Missing required column: '{mean_column}'. "
                "Run add_rolling_means() first."
            )

        df[f"mean_distance_{window}"] = df["close"] - df[mean_column]

    return df


# ============================================================
# PRICE Z-SCORE
# ============================================================


def add_price_zscores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the rolling price Z-score.

    Formula:

        zscore_N =
            (close - rolling_mean_N)
            /
            rolling_std_N

    This expresses price displacement in standard-deviation
    units.

    Example:

        close = 20,100
        rolling mean = 20,000
        rolling std = 50

        zscore = (20,100 - 20,000) / 50
               = +2.0

    Zero standard deviation produces NaN rather than infinity.
    """

    required = {
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    for window in ZSCORE_WINDOWS:
        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"

        if mean_column not in df.columns:
            raise KeyError(
                f"Missing required column: '{mean_column}'. "
                "Run add_rolling_means() first."
            )

        if std_column not in df.columns:
            raise KeyError(
                f"Missing required column: '{std_column}'. Run add_rolling_std() first."
            )

        std = df[std_column].replace(0, np.nan)

        df[f"zscore_{window}"] = (df["close"] - df[mean_column]) / std

    return df


# ============================================================
# LOG-PRICE Z-SCORE
# ============================================================


def add_log_price_zscores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Z-scores using log-price instead of raw price.

    First:

        log_price = log(close)

    Then:

        log_zscore_N =
            (log_price - rolling_mean(log_price))
            /
            rolling_std(log_price)

    Log-price normalization can be useful when investigating
    percentage-based deviations rather than absolute point
    deviations.

    This is kept separate from the normal price Z-score so that
    the research stage can compare both definitions.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    df["log_price"] = np.log(df["close"])

    for window in ZSCORE_WINDOWS:
        mean = df["log_price"].rolling(window).mean()

        std = df["log_price"].rolling(window).std().replace(0, np.nan)

        df[f"log_zscore_{window}"] = (df["log_price"] - mean) / std

    return df


# ============================================================
# Z-SCORE ABSOLUTE MAGNITUDE
# ============================================================


def add_absolute_zscores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate absolute Z-score magnitude.

        abs_zscore_N = abs(zscore_N)

    This removes direction and measures only the magnitude of
    the deviation from the rolling mean.

    Example:

        zscore = -2.3
        abs_zscore = 2.3

    This can later be used to investigate whether larger
    deviations have stronger mean-reversion behavior.
    """

    df = df.copy()

    for window in ZSCORE_WINDOWS:
        column = f"zscore_{window}"

        if column not in df.columns:
            raise KeyError(
                f"Missing required column: '{column}'. Run add_price_zscores() first."
            )

        df[f"abs_zscore_{window}"] = df[column].abs()

    return df


# ============================================================
# Z-SCORE DIRECTION
# ============================================================


def add_zscore_direction(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode whether price is above or below its rolling mean.

        +1 -> positive deviation
         0 -> exactly at mean
        -1 -> negative deviation

    This separates deviation magnitude from deviation direction.
    """

    df = df.copy()

    for window in ZSCORE_WINDOWS:
        column = f"zscore_{window}"

        if column not in df.columns:
            raise KeyError(
                f"Missing required column: '{column}'. Run add_price_zscores() first."
            )

        df[f"zscore_direction_{window}"] = np.sign(df[column])

    return df


# ============================================================
# MASTER FUNCTION
# ============================================================


def add_zscore_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete Z-score feature set.

    Pipeline:

        CLOSE
          |
          +--> rolling mean
          |
          +--> rolling standard deviation
          |
          +--> mean distance
          |
          +--> price Z-score
          |
          +--> absolute Z-score
          |
          +--> Z-score direction
          |
          +--> log-price Z-score
          |
          v
       Z-SCORE FEATURES

    No entry or exit logic is included.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    df = add_rolling_means(df)

    df = add_rolling_std(df)

    df = add_mean_distance(df)

    df = add_price_zscores(df)

    df = add_absolute_zscores(df)

    df = add_zscore_direction(df)

    df = add_log_price_zscores(df)

    return df
