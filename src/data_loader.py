from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validator import validate_dataset
from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)
from src.session_engine import add_session_information
from src.targets import add_future_volatility_targets

def load_data():
    """Load, validate, and prepare the NQ dataset."""

    df = pd.read_csv("data/Dataset_NQ_1min_2022_2025.csv")

    df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])

    validate_dataset(df)

    df["timestamp ET"] = df["timestamp ET"].dt.tz_localize("America/New_York")

    df = add_session_information(df)

    df = add_return_features(df)

    df = add_volatility_features(df)

    df = add_future_volatility_targets(df)

    return df


df = load_data()

returns = df["return"].dropna()

print("\n=== RETURN SANITY CHECK ===")

print("Total observations:", len(df))
print("Valid returns:", len(returns))
print("Missing returns:", df["return"].isna().sum())
print("Infinite returns:", np.isinf(returns).sum())

print("\nMean:", returns.mean())
print("Std:", returns.std())

print("\nQuantiles:")
print(returns.quantile([0.01, 0.05, 0.50, 0.95, 0.99]))

print("\nMinimum:", returns.min())
print("Maximum:", returns.max())


print("\n=== EXTREME RETURNS ===")

extreme_mask = df["return"].abs() > 0.01

extreme = df.loc[
    extreme_mask,
    [
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
        "return",
        "market_period",
    ],
].copy()

extreme["previous_close"] = df["close"].shift(1)[extreme_mask]

extreme["gap_return"] = extreme["open"] / extreme["previous_close"] - 1

extreme["intrabar_return"] = extreme["close"] / extreme["open"] - 1

print(extreme.sort_values("return").to_string(index=False))


print("\n=== VOLATILITY SANITY CHECK ===")

for column in [
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
]:
    print(f"\n{column}")

    print("Missing:", df[column].isna().sum())
    print("Min:", df[column].min())
    print("Median:", df[column].median())
    print("Mean:", df[column].mean())
    print("Max:", df[column].max())


print("\n=== VOLATILITY RATIO SANITY CHECK ===")

for column in [
    "vol_ratio_5_30",
    "vol_ratio_5_60",
]:
    print(f"\n{column}")

    print("Missing:", df[column].isna().sum())
    print("Min:", df[column].min())
    print("25%:", df[column].quantile(0.25))
    print("Median:", df[column].median())
    print("75%:", df[column].quantile(0.75))
    print("Max:", df[column].max())

print("\n=== VARIANCE EXPANSION SANITY CHECK ===")

for column in [
    "variance_ratio_5_30",
    "variance_ratio_5_60",
]:
    print(f"\n{column}")

    print("Missing:", df[column].isna().sum())
    print("Min:", df[column].min())
    print("25%:", df[column].quantile(0.25))
    print("Median:", df[column].median())
    print("75%:", df[column].quantile(0.75))
    print("95%:", df[column].quantile(0.95))
    print("99%:", df[column].quantile(0.99))
    print("Max:", df[column].max())
