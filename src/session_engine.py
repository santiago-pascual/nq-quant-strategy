from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_information(df):

    df["session_date"] = (df["timestamp ET"] - pd.Timedelta(hours=18)).dt.date

    minutes_since_midnight = (df["timestamp ET"].dt.hour * 60) + df[
        "timestamp ET"
    ].dt.minute

    df["market_period"] = np.select(
        [
            (minutes_since_midnight >= 570) & (minutes_since_midnight < 1020),
            (1020 <= minutes_since_midnight) & (minutes_since_midnight < 1080),
        ],
        ["RTH", "BREAK"],
        default="ETH",
    )

    return df
