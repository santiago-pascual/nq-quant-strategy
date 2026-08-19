import pandas as pd

from src.session_engine import add_session_information


def test_session_boundaries():

    timestamps = pd.to_datetime([
        "2025-01-06 09:29:00",
        "2025-01-06 09:30:00",
        "2025-01-06 09:31:00",
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
    "ETH",    # 09:29
    "RTH",    # 09:30
    "RTH",    # 09:31
    "RTH",    # 16:59
    "BREAK",  # 17:00
    "BREAK",  # 17:59
    "ETH",    # 18:00
    "ETH",    # 18:01
    ]

    expected_session_dates = [
    "2025-01-05",  # 09:29
    "2025-01-05",  # 09:30
    "2025-01-05",  # 09:31
    "2025-01-05",  # 16:59
    "2025-01-05",  # 17:00
    "2025-01-05",  # 17:59
    "2025-01-06",  # 18:00
    "2025-01-06",  # 18:01
    ]

    assert df["market_period"].tolist() == expected_periods

    assert (
        df["session_date"]
        .astype(str)
        .tolist()
        == expected_session_dates
        )

    assert df.loc[
        df["market_period"] == "RTH",
        "minutes_since_rth_open"
    ].tolist() == [
        0.0,
        1.0,
        449.0
        ]   

    assert df.loc[
        df["market_period"] == "RTH",
        "minutes_until_rth_close"
    ].tolist() == [
        450.0,
        449.0,
        1.0
        ]

    assert df.loc[
        df["market_period"] == "RTH",
        "rth_progress"
    ].tolist() == [
        0.0,
        1 / 450,
        449 / 450,
    ]

    assert df.loc[
        df["market_period"] != "RTH",
        "minutes_since_rth_open"
    ].isna().all()

    assert df.loc[
            df["market_period"] != "RTH",
            "minutes_until_rth_close"
        ].isna().all()

    assert df.loc[
            df["market_period"] != "RTH",
            "rth_progress"
        ].isna().all()

