from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data


WINDOWS = [5, 10, 15, 30]


# ============================================================
# HELPERS
# ============================================================


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


# ============================================================
# 1. UPSIDE / DOWNSIDE PRESSURE
# ============================================================


def add_directional_pressure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    log_return = df["log_return"]

    upside = log_return.clip(lower=0)

    downside = (-log_return).clip(lower=0)

    for window in WINDOWS:
        df[f"upside_sum_{window}"] = upside.rolling(window).sum()

        df[f"downside_sum_{window}"] = downside.rolling(window).sum()

        df[f"net_pressure_{window}"] = (
            df[f"upside_sum_{window}"] - df[f"downside_sum_{window}"]
        )

        total_movement = df[f"upside_sum_{window}"] + df[f"downside_sum_{window}"]

        df[f"directional_pressure_{window}"] = _safe_divide(
            df[f"net_pressure_{window}"],
            total_movement,
        )

    return df


# ============================================================
# 2. DIRECTIONAL STREAK
# ============================================================


def add_directional_streak_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    returns = df["log_return"]

    direction = np.sign(returns)

    streak = np.zeros(
        len(df),
        dtype=float,
    )

    current = 0.0

    for i, value in enumerate(direction.to_numpy()):
        if np.isnan(value):
            current = 0.0
            streak[i] = np.nan
            continue

        if value > 0:
            if current > 0:
                current += 1.0
            else:
                current = 1.0

        elif value < 0:
            if current < 0:
                current -= 1.0
            else:
                current = -1.0

        else:
            current = 0.0

        streak[i] = current

    df["direction_streak"] = streak

    df["up_streak"] = df["direction_streak"].clip(lower=0)

    df["down_streak"] = -df["direction_streak"].clip(upper=0)

    return df


# ============================================================
# 3. RANGE LOCATION
# ============================================================


def add_range_location_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    close = df["close"]

    for window in WINDOWS:
        rolling_high = close.rolling(window).max()

        rolling_low = close.rolling(window).min()

        range_size = rolling_high - rolling_low

        df[f"close_location_{window}"] = _safe_divide(
            close - rolling_low,
            range_size,
        )

    return df


# ============================================================
# 4. NORMALIZED MOMENTUM
# ============================================================


def add_normalized_momentum_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize momentum using volatility windows that
    actually exist in the core feature engine.

    Existing volatility features:

        realized_vol_5
        realized_vol_15
        realized_vol_30
        realized_vol_60

    Therefore:

        past_return_10 / realized_vol_5
        past_return_15 / realized_vol_15
        past_return_30 / realized_vol_30

    No future information is used.
    """

    df = df.copy()

    normalization_map = {
        10: 5,
        15: 15,
        30: 30,
    }

    for momentum_window, volatility_window in normalization_map.items():
        return_column = f"past_return_{momentum_window}"

        volatility_column = f"realized_vol_{volatility_window}"

        if return_column not in df.columns:
            raise KeyError(f"Missing required column: {return_column}")

        if volatility_column not in df.columns:
            raise KeyError(f"Missing required column: {volatility_column}")

        df[f"normalized_momentum_{momentum_window}"] = _safe_divide(
            df[return_column],
            df[volatility_column],
        )

    return df


# ============================================================
# 5. COMBINED FEATURE ENGINE
# ============================================================


def add_directional_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = add_directional_pressure_features(df)

    df = add_directional_streak_features(df)

    df = add_range_location_features(df)

    df = add_normalized_momentum_features(df)

    return df


# ============================================================
# FEATURE LIST
# ============================================================

DIRECTIONAL_FEATURES = [
    "directional_pressure_5",
    "directional_pressure_10",
    "directional_pressure_15",
    "directional_pressure_30",
    "direction_streak",
    "up_streak",
    "down_streak",
    "close_location_5",
    "close_location_10",
    "close_location_15",
    "close_location_30",
    "normalized_momentum_10",
    "normalized_momentum_15",
    "normalized_momentum_30",
]


# ============================================================
# DIAGNOSTIC
# ============================================================


def main():

    print("=" * 70)
    print("DIRECTIONAL FEATURE ENGINE")
    print("=" * 70)

    df = load_data()

    print(
        "\nInput observations:",
        len(df),
    )

    df = add_directional_features(df)

    print(
        "Output observations:",
        len(df),
    )

    print("\nDirectional features:")

    for feature in DIRECTIONAL_FEATURES:
        print(
            f"{feature:35s}",
            feature in df.columns,
        )

    print("\nFeature preview:")

    print(df[DIRECTIONAL_FEATURES].tail(10))

    print("\nNaN counts:")

    print(df[DIRECTIONAL_FEATURES].isna().sum())


if __name__ == "__main__":
    main()
