from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# MEAN REVERSION — ORNSTEIN-UHLENBECK FEATURES
# ============================================================
#
# PURPOSE
# -------
# The Ornstein-Uhlenbeck (OU) process is a classical model
# for a variable that tends to move back toward an equilibrium.
#
# A continuous-time OU process can be written as:
#
#     dX_t = theta * (mu - X_t) * dt + sigma * dW_t
#
# where:
#
#     mu    = equilibrium / long-run mean
#     theta = speed of mean reversion
#     sigma = noise / diffusion
#
# For market research we do NOT assume that NQ is literally
# an OU process.
#
# Instead, we use OU-related statistics as diagnostic tools:
#
#     "Does recent price behavior exhibit characteristics
#      consistent with mean reversion?"
#
# IMPORTANT
# ---------
# This module does NOT:
#
#     - generate entries
#     - generate exits
#     - choose thresholds
#     - optimize parameters
#
# It only creates descriptive statistics.
#
# ============================================================
#
# NO LOOK-AHEAD
# ------------
#
# Every rolling estimate uses observations available up to
# the current bar.
#
# Future observations are never included.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

OU_WINDOWS = (30, 60, 120)


# ============================================================
# LAG-1 AUTOCORRELATION
# ============================================================


def add_autocorrelation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate rolling lag-1 autocorrelation of price changes.

    For a window N:

        autocorr_N =
            Corr(
                ΔP_t,
                ΔP_(t-1)
            )

    Interpretation:

        positive autocorrelation
            -> successive changes tend to have the same sign

        negative autocorrelation
            -> successive changes tend to alternate

    Negative short-horizon autocorrelation can be consistent
    with short-term mean-reverting behavior.

    This is evidence to investigate, NOT a trading signal.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    price_change = df["close"].diff()

    for window in OU_WINDOWS:
        df[f"autocorrelation_{window}"] = price_change.rolling(window).corr(
            price_change.shift(1)
        )

    return df


# ============================================================
# AR(1) COEFFICIENT
# ============================================================


def add_ar1_coefficient(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate the rolling AR(1) coefficient of price changes.

    We model:

        ΔP_t = alpha + beta * ΔP_(t-1) + epsilon_t

    The coefficient beta is estimated using rolling covariance
    and variance:

        beta =
            Cov(X_t, X_(t-1))
            /
            Var(X_(t-1))

    Interpretation:

        beta > 0
            -> persistence

        beta < 0
            -> reversal tendency

    This is mathematically related to lag-1 autocorrelation
    when the relevant means are handled consistently.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    change = df["close"].diff()
    lagged_change = change.shift(1)

    for window in OU_WINDOWS:
        covariance = change.rolling(window).cov(lagged_change)

        variance = lagged_change.rolling(window).var()

        variance = variance.replace(
            0,
            np.nan,
        )

        df[f"ar1_coefficient_{window}"] = covariance / variance

    return df


# ============================================================
# MEAN REVERSION SPEED
# ============================================================


def add_mean_reversion_speed(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the AR(1) coefficient into a discrete-time
    mean-reversion speed estimate.

    For an AR(1) process:

        X_t = phi * X_(t-1) + epsilon_t

    with:

        0 < phi < 1

    the process exhibits persistence toward an equilibrium.

    A common continuous-time approximation is:

        theta = -ln(phi)

    However, this transformation is meaningful only for:

        0 < phi < 1

    Therefore values outside this region are returned as NaN.

    This prevents mathematically invalid logarithms and avoids
    silently interpreting explosive or oscillatory processes as
    conventional OU mean reversion.
    """

    df = df.copy()

    for window in OU_WINDOWS:
        column = f"ar1_coefficient_{window}"

        if column not in df.columns:
            raise KeyError(
                f"Missing required column: '{column}'. Run add_ar1_coefficient() first."
            )

        phi = df[column]

        valid_phi = phi.where((phi > 0) & (phi < 1))

        df[f"mean_reversion_speed_{window}"] = -np.log(valid_phi)

    return df


# ============================================================
# HALF-LIFE
# ============================================================


def add_half_life(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate the half-life of mean reversion.

    If:

        theta = -ln(phi)

    then the approximate half-life is:

        half_life = ln(2) / theta

    Equivalently:

        half_life = -ln(2) / ln(phi)

    The result is expressed in bars.

    Example:

        half_life = 10

    means that, under the estimated AR(1)/OU approximation,
    deviations decay by approximately 50% over 10 bars.

    IMPORTANT:

    This is an estimated statistical property, not a guarantee
    that price will return within that number of bars.
    """

    df = df.copy()

    for window in OU_WINDOWS:
        speed_column = f"mean_reversion_speed_{window}"

        if speed_column not in df.columns:
            raise KeyError(
                f"Missing required column: '{speed_column}'. "
                "Run add_mean_reversion_speed() first."
            )

        speed = df[speed_column]

        df[f"half_life_{window}"] = np.log(2.0) / speed

    return df


# ============================================================
# OU RESIDUAL / EQUILIBRIUM DISTANCE
# ============================================================


def add_ou_residual(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the residual from the rolling equilibrium proxy.

    For this research layer we use the rolling mean as the
    equilibrium proxy:

        residual_N =
            close - rolling_mean_N

    This intentionally overlaps conceptually with the Z-score
    feature.

    The purpose is to provide a common variable for OU-oriented
    research without embedding a strategy-specific equilibrium
    estimator.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    for window in OU_WINDOWS:
        mean = df["close"].rolling(window).mean()

        df[f"ou_residual_{window}"] = df["close"] - mean

    return df


# ============================================================
# OU RESIDUAL NORMALIZED
# ============================================================


def add_normalized_ou_residual(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the OU residual by rolling price volatility.

        normalized_residual_N =
            residual_N / rolling_std_N

    This produces a dimensionless measure of how large the
    deviation is relative to the recent price dispersion.

    It is closely related to a price Z-score but is retained
    here because OU research operates naturally in terms of
    deviations from an equilibrium.
    """

    df = df.copy()

    for window in OU_WINDOWS:
        residual_column = f"ou_residual_{window}"

        if residual_column not in df.columns:
            raise KeyError(
                f"Missing required column: "
                f"'{residual_column}'. "
                "Run add_ou_residual() first."
            )

        rolling_std = df["close"].rolling(window).std().replace(0, np.nan)

        df[f"normalized_ou_residual_{window}"] = df[residual_column] / rolling_std

    return df


# ============================================================
# MASTER FUNCTION
# ============================================================


def add_ou_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete OU-related feature set.

    Pipeline:

        CLOSE
          |
          +--> price changes
          |
          +--> autocorrelation
          |
          +--> AR(1) coefficient
          |
          +--> mean-reversion speed
          |
          +--> half-life
          |
          +--> equilibrium residual
          |
          +--> normalized residual
          |
          v
       OU FEATURES

    No strategy logic is included.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    df = add_autocorrelation(df)

    df = add_ar1_coefficient(df)

    df = add_mean_reversion_speed(df)

    df = add_half_life(df)

    df = add_ou_residual(df)

    df = add_normalized_ou_residual(df)

    return df
