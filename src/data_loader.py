from __future__ import annotations

import numpy as np
import pandas as pd

from session_engine import (
    add_session_information)

from data_validator import (
    check_missing_values,
    check_ohlc_consistency,
    check_negative_volume,
    check_duplicates,
    validate_dataset,
    check_timestamp_order
)

df = pd.read_csv("data/Dataset_NQ_1min_2022_2025.csv")

df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])

report = validate_dataset(df)

rows_per_day = df["timestamp ET"].dt.date.value_counts().sort_index()

df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])
df["timestamp ET"] = df["timestamp ET"].dt.tz_localize("America/New_York")

df = add_session_information(df)

print("\n=== 09:29 → 09:30 ===")
print(
    df[
        (df["timestamp ET"].dt.hour == 9) &
        (df["timestamp ET"].dt.minute.isin([29, 30, 31]))
    ][["timestamp ET", "session_date", "market_period"]]
)

print("\n=== 16:59 → 17:00 ===")
print(
    df[
        (df["timestamp ET"].dt.hour == 16) &
        (df["timestamp ET"].dt.minute == 59)
        |
        (
            (df["timestamp ET"].dt.hour == 17) &
            (df["timestamp ET"].dt.minute == 0)
        )
    ][["timestamp ET", "session_date", "market_period"]]
)

print("\n=== 17:59 → 18:00 ===")
print(
    df[
        (
            (df["timestamp ET"].dt.hour == 17) &
            (df["timestamp ET"].dt.minute == 59)
        )
        |
        (
            (df["timestamp ET"].dt.hour == 18) &
            (df["timestamp ET"].dt.minute == 0)
        )
    ][["timestamp ET", "session_date", "market_period"]]
)

print("\n=== 17:59 / 18:00 DATA AVAILABILITY ===")

print(
    df[
        (
            (df["timestamp ET"].dt.hour == 17) &
            (df["timestamp ET"].dt.minute == 59)
        )
        |
        (
            (df["timestamp ET"].dt.hour == 18) &
            (df["timestamp ET"].dt.minute == 0)
        )
    ][["timestamp ET", "session_date", "market_period"]]
)

print("\n=== SESSION OPEN ===")

print(
    df[
        (df["timestamp ET"] >= "2022-12-26 17:55:00-05:00") &
        (df["timestamp ET"] <= "2022-12-26 18:10:00-05:00")
    ][["timestamp ET", "session_date", "market_period"]]
)

print(report)

print(df["timestamp ET"].min())

print(df["timestamp ET"].max())

print(df["timestamp ET"].dt.date.nunique())

print(df["timestamp ET"].dt.hour.min())

print(df["timestamp ET"].dt.hour.max())
