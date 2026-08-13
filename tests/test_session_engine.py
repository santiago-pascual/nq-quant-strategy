import pandas as pd

from src.session_engine import add_session_information


def test_session_boundaries():

    timestamps = pd.to_datetime([
        "2025-01-06 09:29:00",
        "2025-01-06 09:30:00",
        "2025-01-06 16:59:00",
        "2025-01-06 17:00:00",
        "2025-01-06 17:59:00",
        "2025-01-06 18:00:00",
        "2025-01-06 18:01:00",
    ]).tz_localize("America/New_York")

    df = pd.DataFrame({
        "timestamp ET": timestamps
    })

    df = add_session_information(df)

    expected_periods = [
        "ETH",
        "RTH",
        "RTH",
        "BREAK",
        "BREAK",
        "ETH",
        "ETH",
    ]

    expected_session_dates = [
        "2025-01-05",
        "2025-01-05",
        "2025-01-05",
        "2025-01-05",
        "2025-01-05",
        "2025-01-06",
        "2025-01-06",
    ]

    assert df["market_period"].tolist() == expected_periods

    assert (
        df["session_date"]
        .astype(str)
        .tolist()
        == expected_session_dates
    )