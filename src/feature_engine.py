from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(df):

    df["return"] = df["close"].pct_change()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    return df


def add_volatility_features(df):

    df["realized_vol_5"] = np.sqrt((df["log_return"] ** 2).rolling(5).sum())

    df["realized_vol_15"] = np.sqrt((df["log_return"] ** 2).rolling(15).sum())

    df["realized_vol_30"] = np.sqrt((df["log_return"] ** 2).rolling(30).sum())

    df["realized_vol_60"] = np.sqrt((df["log_return"] ** 2).rolling(60).sum())

    df["vol_ratio_5_30"] = df["realized_vol_5"] / df["realized_vol_30"]

    df["vol_ratio_5_60"] = df["realized_vol_5"] / df["realized_vol_60"]

    return df
