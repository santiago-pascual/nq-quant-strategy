from __future__ import annotations

import pandas as pd

from src.data_loader import load_data
from src.targets import add_future_volatility_targets


def analyze_volatility_relationship(
    df,
    feature,
    target,
    quantiles=5,
):

    analysis = df[[feature, target]].dropna().copy()

    analysis["feature_quantile"] = pd.qcut(
        analysis[feature],
        q=quantiles,
        labels=False,
    )

    result = analysis.groupby("feature_quantile", observed=True)[target].agg(
        observations="count",
        mean="mean",
        median="median",
    )

    return result


def analyze_volatility_by_session(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> pd.DataFrame:

    analysis = df[["market_period", feature, target]].dropna().copy()

    results = []

    for session in ["RTH", "ETH"]:
        session_data = analysis[analysis["market_period"] == session].copy()

        if session_data.empty:
            continue

        session_data["feature_quantile"] = pd.qcut(
            session_data[feature],
            q=5,
            labels=False,
            duplicates="drop",
        )

        grouped = (
            session_data.groupby("feature_quantile", observed=True)[target]
            .agg(["count", "mean", "median"])
            .reset_index()
        )

        grouped["market_period"] = session
        grouped["feature"] = feature
        grouped["target"] = target

        results.append(grouped)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def calculate_correlations(
    df,
    feature,
    target,
):

    analysis = df[[feature, target]].dropna()

    pearson = analysis[feature].corr(
        analysis[target],
        method="pearson",
    )

    spearman = analysis[feature].corr(analysis[target], method="spearman")

    return pearson, spearman


def main():

    df = load_data()
    df = add_future_volatility_targets(df)

    features = [
        "variance_ratio_5_30",
        "variance_ratio_5_60",
    ]

    targets = [
        "future_vol_5",
        "future_vol_15",
        "future_vol_30",
    ]

    for feature in features:
        for target in targets:
            result = analyze_volatility_by_session(
                df=df,
                feature=feature,
                target=target,
            )

            print("\n========================================")
            print(f"{feature} → {target}")
            print("========================================")

            print(result.to_string(index=False))


if __name__ == "__main__":
    main()
