from __future__ import annotations

import pandas as pd

from src.data_loader import load_data


PAST_WINDOWS = [1, 3, 5, 10, 15, 30]
FUTURE_WINDOWS = [5, 15, 30]


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for window in PAST_WINDOWS:
        df[f"past_return_{window}"] = df["log_return"].shift(1).rolling(window).sum()

    return df


def analyze_momentum_relationship(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> pd.DataFrame:

    analysis = (
        df.loc[
            df["market_period"] == "RTH",
            [feature, target],
        ]
        .dropna()
        .copy()
    )

    analysis["momentum_quantile"] = pd.qcut(
        analysis[feature],
        q=5,
        labels=False,
        duplicates="drop",
    )

    result = analysis.groupby("momentum_quantile", observed=True)[target].agg(
        observations="count",
        mean="mean",
        median="median",
    )

    return result


def main():

    df = load_data()

    df = add_momentum_features(df)

    print("\n=== RTH RETURN AUTOCORRELATION ===")

    rth_returns = df.loc[
        df["market_period"] == "RTH",
        "log_return",
    ].dropna()

    for lag in [1, 2, 3, 5, 10, 15, 30]:
        correlation = rth_returns.autocorr(lag=lag)

        print(f"Lag {lag:>2}: {correlation:.8f}")

    print("\n" + "=" * 60)
    print("MOMENTUM → FUTURE RETURN")
    print("=" * 60)

    for past_window in PAST_WINDOWS:
        feature = f"past_return_{past_window}"

        for future_window in FUTURE_WINDOWS:
            target = f"future_return_{future_window}"

            print(f"\n--- {feature} → {target} ---")

            result = analyze_momentum_relationship(
                df=df,
                feature=feature,
                target=target,
            )

            print(result)


if __name__ == "__main__":
    main()
