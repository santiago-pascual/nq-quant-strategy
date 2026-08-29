from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.targets import add_future_return_targets

TRAIN_END = "2024-12-31 23:59:59"
OOS_START = "2025-01-01 00:00:00"

MOMENTUM_FEATURES = [
    "past_return_10",
    "past_return_15",
    "past_return_30",
]

TARGETS = [
    "future_return_5",
    "future_return_15",
    "future_return_30",
]


def calculate_train_quantile_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:
    """
    Calculate quintile boundaries using TRAIN data only.

    These boundaries are then frozen and applied to OOS data.
    """

    values = train[feature].dropna()

    bins = values.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()

    bins = np.unique(bins)

    if len(bins) < 2:
        raise ValueError(f"Not enough unique values to create bins for {feature}.")

    return bins


def assign_frozen_quantiles(
    df: pd.DataFrame,
    feature: str,
    bins: np.ndarray,
) -> pd.Series:
    """
    Assign observations to quantile bins using boundaries
    calculated exclusively from TRAIN data.
    """

    bin_edges = bins.tolist()

    return pd.cut(
        df[feature],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )


def analyze_oos_relationship(
    oos: pd.DataFrame,
    feature: str,
    target: str,
    bins: np.ndarray,
) -> pd.DataFrame:
    """
    Analyze the relationship between a momentum feature and
    future returns using frozen TRAIN quantile boundaries.
    """

    analysis = oos[[feature, target]].dropna().copy()

    analysis["momentum_quantile"] = assign_frozen_quantiles(
        analysis,
        feature,
        bins,
    )

    analysis = analysis.dropna(subset=["momentum_quantile"])

    analysis["momentum_quantile"] = analysis["momentum_quantile"].astype(int)

    result = analysis.groupby("momentum_quantile")[target].agg(
        observations="count",
        mean="mean",
        median="median",
    )

    return result


def main() -> None:

    print("=" * 60)
    print("DIRECTION OOS VALIDATION")
    print("=" * 60)

    df = load_data()

    # ---------------------------------------------------------
    # TARGETS
    # ---------------------------------------------------------

    df = add_future_return_targets(df)

    # ---------------------------------------------------------
    # TIMEZONE / SORTING
    # ---------------------------------------------------------

    df = df.sort_values("timestamp ET").copy()

    # ---------------------------------------------------------
    # RTH ONLY
    # ---------------------------------------------------------

    rth = df[df["market_period"] == "RTH"].copy()

    print("\n=== RTH DATA ===")
    print("Observations:", len(rth))
    print("Start:", rth["timestamp ET"].min())
    print("End:", rth["timestamp ET"].max())

    # ---------------------------------------------------------
    # TRAIN / OOS SPLIT
    # ---------------------------------------------------------

    train = rth[
        rth["timestamp ET"]
        <= pd.Timestamp(
            TRAIN_END,
            tz="America/New_York",
        )
    ].copy()

    oos = rth[
        rth["timestamp ET"]
        >= pd.Timestamp(
            OOS_START,
            tz="America/New_York",
        )
    ].copy()

    print("\n=== DATA SPLIT ===")
    print("Train observations:", len(train))
    print("Train start:", train["timestamp ET"].min())
    print("Train end:", train["timestamp ET"].max())

    print("\nOOS observations:", len(oos))
    print("OOS start:", oos["timestamp ET"].min())
    print("OOS end:", oos["timestamp ET"].max())

    # ---------------------------------------------------------
    # MOMENTUM FEATURES
    # ---------------------------------------------------------

    for feature in MOMENTUM_FEATURES:
        print("\n" + "=" * 60)
        print(f"MOMENTUM FEATURE: {feature}")
        print("=" * 60)

        # -----------------------------------------------------
        # TRAIN QUANTILES
        # -----------------------------------------------------

        bins = calculate_train_quantile_bins(
            train=train,
            feature=feature,
        )

        print("\nTrain quantile boundaries:")

        for i, value in enumerate(bins):
            print(f"Boundary {i}: {value:.10f}")

        # -----------------------------------------------------
        # OOS TARGET ANALYSIS
        # -----------------------------------------------------

        for target in TARGETS:
            print(f"\n--- {feature} -> {target} (OOS) ---")

            result = analyze_oos_relationship(
                oos=oos,
                feature=feature,
                target=target,
                bins=bins,
            )

            print(result.to_string())

            # -------------------------------------------------
            # Q5 - Q1
            # -------------------------------------------------

            if 0 in result.index and 4 in result.index:
                q1 = result.loc[0, "mean"]
                q5 = result.loc[4, "mean"]

                spread = q5 - q1

                print(f"\nQ5 - Q1 mean return: {spread:.10f}")

                if q1 != 0:
                    ratio = q5 / q1
                    print(f"Q5 / Q1 mean ratio: {ratio:.4f}")

            # -------------------------------------------------
            # SPEARMAN CORRELATION
            # -------------------------------------------------

            correlation_data = oos[[feature, target]].dropna()

            if len(correlation_data) > 1:
                spearman = correlation_data[feature].corr(
                    correlation_data[target],
                    method="spearman",
                )

                print(f"Spearman correlation: {spearman:.10f}")


if __name__ == "__main__":
    main()
