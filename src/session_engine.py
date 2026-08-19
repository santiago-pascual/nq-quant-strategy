from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_information(df):

    # ---------------------------------------------------------
    # 1. Determine the trading session date
    # ---------------------------------------------------------

    df["session_date"] = (
        df["timestamp ET"] - pd.Timedelta(hours=18)
    ).dt.date

    # ---------------------------------------------------------
    # 2. Convert timestamp into minutes since midnight
    # ---------------------------------------------------------

    minutes_since_midnight = (
        df["timestamp ET"].dt.hour * 60
        + df["timestamp ET"].dt.minute
    )

    # ---------------------------------------------------------
    # 3. Classify the market period
    # ---------------------------------------------------------

    df["market_period"] = np.select(
        [
            (minutes_since_midnight >= 570)
            & (minutes_since_midnight < 1020),

            (minutes_since_midnight >= 1020)
            & (minutes_since_midnight < 1080),
        ],
        [
            "RTH",
            "BREAK",
        ],
        default="ETH",
    )

    # ---------------------------------------------------------
    # 4. Calculate time relative to RTH
    # ---------------------------------------------------------

    rth_mask = df["market_period"] == "RTH"

    df["minutes_since_rth_open"] = np.where(
        rth_mask,
        minutes_since_midnight - 570,
        np.nan,
    )

    df["minutes_until_rth_close"] = np.where(
        rth_mask,
        1020 - minutes_since_midnight,
        np.nan,
    )

    df["rth_progress"] = np.where(
        rth_mask,
        (minutes_since_midnight - 570) / 450,
        np.nan,
    )

    return df