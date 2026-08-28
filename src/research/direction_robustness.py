from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import HMM_FEATURES, VolatilityRegimeModel

# ============================================================
# CONFIGURATION
# ============================================================

MOMENTUM_FEATURES = [
    "past_return_10",
    "past_return_15",
    "past_return_30",
]

TARGETS = {
    5: "future_return_5",
    15: "future_return_15",
    30: "future_return_30",
}

N_STATES = 3
N_QUANTILES = 5
RANDOM_STATE = 42

BOOTSTRAP_ITERATIONS = 2000

TRAIN_END = pd.Timestamp(
    "2024-12-31 16:59:00",
    tz="America/New_York",
)

OOS_START = pd.Timestamp(
    "2025-01-02 09:30:00",
    tz="America/New_York",
)


# ============================================================
# QUANTILES
# ============================================================


def calculate_train_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:

    values = train[feature].dropna()

    bins = np.quantile(
        values,
        np.linspace(
            0.0,
            1.0,
            N_QUANTILES + 1,
        ),
    )

    bins = np.unique(bins)

    if len(bins) != N_QUANTILES + 1:
        raise ValueError(
            f"Could not create {N_QUANTILES} unique quantile bins for {feature}."
        )

    return bins


def apply_quantiles(
    df: pd.DataFrame,
    feature: str,
    bins: np.ndarray,
) -> pd.Series:
    bin_edges = bins.tolist()

    return pd.cut(
        df[feature],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )


# ============================================================
# NON-OVERLAPPING SAMPLE
# ============================================================


def create_nonoverlapping_sample(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:

    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    # Primary deterministic stream.
    return df.iloc[0::horizon].copy()


# ============================================================
# BASIC STATISTICS
# ============================================================


def calculate_stats(
    returns: pd.Series,
) -> dict:

    values = returns.dropna().to_numpy()

    n = len(values)

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    wins = values[values > 0]
    losses = values[values < 0]

    win_rate = len(wins) / n

    gross_profit = wins.sum() if len(wins) else 0.0

    gross_loss = abs(losses.sum()) if len(losses) else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "n": n,
        "mean": values.mean(),
        "median": np.median(values),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


# ============================================================
# BOOTSTRAP
# ============================================================


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:

    values = np.asarray(values)

    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(
            iterations,
            len(values),
        ),
    )

    bootstrap_means = values[indices].mean(axis=1)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return lower, upper


def bootstrap_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:

    a = np.asarray(a)
    b = np.asarray(b)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    a_idx = rng.integers(
        0,
        len(a),
        size=(
            iterations,
            len(a),
        ),
    )

    b_idx = rng.integers(
        0,
        len(b),
        size=(
            iterations,
            len(b),
        ),
    )

    a_means = a[a_idx].mean(axis=1)

    b_means = b[b_idx].mean(axis=1)

    differences = a_means - b_means

    lower, upper = np.percentile(
        differences,
        [2.5, 97.5],
    )

    return lower, upper


# ============================================================
# PERMUTATION TEST
# ============================================================


def permutation_p_value(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> float:

    a = np.asarray(a)
    b = np.asarray(b)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    observed = a.mean() - b.mean()

    combined = np.concatenate([a, b])

    rng = np.random.default_rng(seed)

    count = 0

    for _ in range(iterations):
        shuffled = rng.permutation(combined)

        new_a = shuffled[: len(a)]
        new_b = shuffled[len(a) :]

        difference = new_a.mean() - new_b.mean()

        if abs(difference) >= abs(observed):
            count += 1

    return (count + 1) / (iterations + 1)


# ============================================================
# DIRECTION
# ============================================================


def direction_from_return(
    values: pd.Series,
) -> pd.Series:
    """
    Convert realized forward return into:

        +1 = LONG
        -1 = SHORT
         0 = FLAT

    This function is used only for descriptive statistics.
    It does not create a trading signal.
    """

    result = pd.Series(
        0,
        index=values.index,
        dtype=int,
    )

    result.loc[values > 0] = 1

    result.loc[values < 0] = -1

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("DIRECTION × HMM — LONG / SHORT ROBUSTNESS")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    required = (
        [
            "timestamp ET",
            "market_period",
        ]
        + HMM_FEATURES
        + MOMENTUM_FEATURES
        + list(TARGETS.values())
    )

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    timestamp = pd.to_datetime(df["timestamp ET"])

    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize("America/New_York")

    else:
        timestamp = timestamp.dt.tz_convert("America/New_York")

    # ========================================================
    # RTH
    # ========================================================

    rth = df.loc[df["market_period"] == "RTH"].copy()

    rth_timestamp = timestamp.loc[rth.index]

    # ========================================================
    # TRAIN / OOS
    # ========================================================

    train = rth.loc[rth_timestamp <= TRAIN_END].copy()

    oos = rth.loc[rth_timestamp >= OOS_START].copy()

    print("\n=== DATA SPLIT ===")

    print(
        "Train:",
        len(train),
    )

    print(
        "OOS:",
        len(oos),
    )

    # ========================================================
    # HMM
    # ========================================================

    print("\n=== FITTING HMM ===")

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    print(
        "Converged:",
        model.model.monitor_.converged,
    )

    print(
        "Iterations:",
        model.model.monitor_.iter,
    )

    train["hmm_state"] = model.predict_states(train)

    oos["hmm_state"] = model.predict_states(oos)

    # ========================================================
    # RESULTS
    # ========================================================

    rows = []

    for feature in MOMENTUM_FEATURES:
        print("\n" + "#" * 70)
        print(f"FEATURE: {feature}")
        print("#" * 70)

        bins = calculate_train_bins(
            train,
            feature,
        )

        oos["momentum_quantile"] = apply_quantiles(
            oos,
            feature,
            bins,
        )

        for horizon, target in TARGETS.items():
            sampled = create_nonoverlapping_sample(
                oos,
                horizon,
            )

            print("\n" + "-" * 70)
            print(f"{feature} -> {target}")
            print(
                "Non-overlapping observations:",
                len(sampled),
            )
            print("-" * 70)

            for state in range(N_STATES):
                state_data = sampled.loc[sampled["hmm_state"] == state]

                # ------------------------------------------------
                # Q1 = LOW MOMENTUM
                # Q5 = HIGH MOMENTUM
                # ------------------------------------------------

                q1 = state_data.loc[
                    state_data["momentum_quantile"] == 0,
                    target,
                ].dropna()

                q5 = state_data.loc[
                    state_data["momentum_quantile"] == 4,
                    target,
                ].dropna()

                q1_values = q1.to_numpy()
                q5_values = q5.to_numpy()

                q1_stats = calculate_stats(q1)

                q5_stats = calculate_stats(q5)

                # ------------------------------------------------
                # LONG = positive forward returns
                # SHORT = negative forward returns
                #
                # We evaluate Q5 and Q1 separately.
                # ------------------------------------------------

                q5_long_stats = calculate_stats(q5)

                # ------------------------------------------------
                # Q1 SHORT
                #
                # Negate returns so that:
                #
                # profitable short = positive return
                # ------------------------------------------------

                q1_short_stats = calculate_stats(-q1)

                # ------------------------------------------------
                # Q5 - Q1
                # ------------------------------------------------

                spread = q5_values.mean() - q1_values.mean()

                ci_low, ci_high = bootstrap_difference_ci(
                    q5_values,
                    q1_values,
                )

                p_value = permutation_p_value(
                    q5_values,
                    q1_values,
                )

                # ------------------------------------------------
                # PRINT
                # ------------------------------------------------

                print(f"\nSTATE {state}")

                print(f"Q1 observations: {len(q1)}")

                print(f"Q5 observations: {len(q5)}")

                print(f"Q1 mean: {q1_stats['mean']:.10f}")

                print(f"Q5 mean: {q5_stats['mean']:.10f}")

                print(f"Q5 - Q1: {spread:.10f}")

                print(
                    "95% bootstrap CI:",
                    f"[{ci_low:.10f}, {ci_high:.10f}]",
                )

                print(f"Permutation p-value: {p_value:.6f}")

                print("\nQ5 LONG:")

                print(f"Win rate: {q5_long_stats['win_rate']:.4f}")

                print(f"Profit factor: {q5_long_stats['profit_factor']:.4f}")

                print("\nQ1 SHORT:")

                print(f"Win rate: {q1_short_stats['win_rate']:.4f}")

                print(f"Profit factor: {q1_short_stats['profit_factor']:.4f}")

                if ci_low > 0:
                    print("RESULT: SIGNIFICANT POSITIVE Q5-Q1 EFFECT")

                elif ci_high < 0:
                    print("RESULT: SIGNIFICANT NEGATIVE Q5-Q1 EFFECT")

                else:
                    print("RESULT: CI CROSSES ZERO")

                rows.append(
                    {
                        "feature": feature,
                        "horizon": horizon,
                        "state": state,
                        "q1_n": len(q1),
                        "q5_n": len(q5),
                        "q1_mean": q1_stats["mean"],
                        "q5_mean": q5_stats["mean"],
                        "q5_minus_q1": spread,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                        "q5_long_pf": q5_long_stats["profit_factor"],
                        "q1_short_pf": q1_short_stats["profit_factor"],
                        "q5_long_win_rate": q5_long_stats["win_rate"],
                        "q1_short_win_rate": q1_short_stats["win_rate"],
                    }
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    results = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(results.to_string(index=False))

    # ========================================================
    # CANDIDATE LONG CONDITIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("CANDIDATE LONG CONDITIONS")
    print("=" * 70)

    long_candidates = results.loc[
        (results["q5_mean"] > 0)
        & (results["q5_long_pf"] > 1.0)
        & (results["ci_low"] > 0)
    ].sort_values(
        "q5_minus_q1",
        ascending=False,
    )

    if long_candidates.empty:
        print("No statistically robust LONG candidates.")

    else:
        print(long_candidates.to_string(index=False))

    # ========================================================
    # CANDIDATE SHORT CONDITIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("CANDIDATE SHORT CONDITIONS")
    print("=" * 70)

    short_candidates = results.loc[
        (results["q1_mean"] < 0)
        & (results["q1_short_pf"] > 1.0)
        & (results["ci_high"] < 0)
    ].sort_values(
        "q5_minus_q1",
        ascending=True,
    )

    if short_candidates.empty:
        print("No statistically robust SHORT candidates.")

    else:
        print(short_candidates.to_string(index=False))

    print("\n" + "=" * 70)
    print("LONG / SHORT ROBUSTNESS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
