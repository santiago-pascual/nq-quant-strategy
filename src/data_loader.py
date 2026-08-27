from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validator import validate_dataset
from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)
from src.session_engine import add_session_information
from src.targets import (
    add_future_return_targets,
    add_future_volatility_targets,
)


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

    df = add_future_return_targets(df)

    return df


def main():

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


if __name__ == "__main__":
    main()
