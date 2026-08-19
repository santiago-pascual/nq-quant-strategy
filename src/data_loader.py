from __future__ import annotations

import pandas as pd

from session_engine import add_session_information

from data_validator import validate_dataset


def load_data():
    """
    Load, validate, and prepare the NQ DataSet.
    """

    df = pd.read_csv("data/Dataset_NQ_1min_2022_2025.csv")

    df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])

    validate_dataset(df)

    df["timestamp ET"] = (
        df["timestamp ET"].dt.tz_localize("America/New_York")
    )
    df = add_session_information(df)

    return df