from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_volatility_targets(df):

    squared_returns = df["log_return"] ** 2

    future_variance_5 = (
        squared_returns.shift(-1)
        + squared_returns.shift(-2)
        + squared_returns.shift(-3)
        + squared_returns.shift(-4)
        + squared_returns.shift(-5)
    )

    future_variance_15 = sum(squared_returns.shift(-i) for i in range(1, 16))

    future_variance_30 = sum(squared_returns.shift(-i) for i in range(1, 31))

    df["future_vol_5"] = np.sqrt(future_variance_5)

    df["future_vol_15"] = np.sqrt(future_variance_15)

    df["future_vol_30"] = np.sqrt(future_variance_30)

    return df


def add_future_return_targets(df):

    df["future_return_5"] = df["log_return"].shift(-1).rolling(5).sum().shift(-4)

    df["future_return_15"] = df["log_return"].shift(-1).rolling(15).sum().shift(-14)

    df["future_return_30"] = df["log_return"].shift(-1).rolling(30).sum().shift(-29)

    return df
