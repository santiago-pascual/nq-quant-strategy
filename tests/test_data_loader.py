from src.data_loader import load_data


def test_load_data():

    df = load_data()

    assert not df.empty

    assert "timestamp ET" in df.columns
    assert "session_date" in df.columns
    assert "market_period" in df.columns

    assert str(df["timestamp ET"].dtype).startswith("datetime64")

    assert df["market_period"].isin(
        ["ETH", "RTH", "BREAK"]
    ).all()