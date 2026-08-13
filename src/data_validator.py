def check_missing_values(df):
    return df.isna().sum()


def check_ohlc_consistency(df):
    errors = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    return errors.sum()


def check_duplicates(df):
    return df.duplicated().sum()


def check_negative_volume(df):
    return (df["volume"] < 0).sum()


def validate_dataset(df):
    report = {
        "missing_values": check_missing_values(df),
        "ohlc_errors": check_ohlc_consistency(df),
        "duplicates": check_duplicates(df),
        "negative_volume": check_negative_volume(df),
        "timestamps_ordered": check_timestamp_order(df),
    }

    return report


def check_timestamp_order(df):
    return df["timestamp ET"].is_monotonic_increasing

