from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return and momentum features.

    All features use information available at or before the
    current observation. No future information is used.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # LOG RETURN
    # ---------------------------------------------------------

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Simple close-to-close return
    df["return"] = (df["close"] / df["close"].shift(1)) - 1

    # ---------------------------------------------------------
    # LONG / SHORT RETURN REPRESENTATION
    # ---------------------------------------------------------

    df["long_return"] = df["return"]

    df["short_return"] = -df["return"]

    # ---------------------------------------------------------
    # PAST RETURN / MOMENTUM
    # ---------------------------------------------------------

    for window in [1, 3, 5, 10, 15, 30]:
        df[f"past_return_{window}"] = df["log_return"].rolling(window).sum()

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realized volatility and volatility-regime features.

    Volatility is calculated exclusively from historical
    log returns.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # REALIZED VOLATILITY
    # ---------------------------------------------------------

    for window in [5, 15, 30, 60]:
        df[f"realized_vol_{window}"] = np.sqrt(
            (df["log_return"] ** 2).rolling(window).sum()
        )

    # ---------------------------------------------------------
    # VOLATILITY RATIOS
    # ---------------------------------------------------------

    df["vol_ratio_5_30"] = df["realized_vol_5"] / df["realized_vol_30"]

    df["vol_ratio_5_60"] = df["realized_vol_5"] / df["realized_vol_60"]

    # ---------------------------------------------------------
    # ROLLING VARIANCE
    # ---------------------------------------------------------

    squared_returns = df["log_return"] ** 2

    df["variance_5"] = squared_returns.rolling(5).mean()

    df["variance_30"] = squared_returns.rolling(30).mean()

    df["variance_60"] = squared_returns.rolling(60).mean()

    # ---------------------------------------------------------
    # VARIANCE RATIOS
    # ---------------------------------------------------------

    df["variance_ratio_5_30"] = df["variance_5"] / df["variance_30"]

    df["variance_ratio_5_60"] = df["variance_5"] / df["variance_60"]

    return df
