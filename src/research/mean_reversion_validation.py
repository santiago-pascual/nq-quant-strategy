from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_END = pd.Timestamp("2024-12-31 23:59:00", tz="America/New_York")

FEATURES = [
    "past_return_5",
    "past_return_10",
    "past_return_15",
    "past_return_30",
]

TARGETS = [
    "future_return_5",
    "future_return_15",
    "future_return_30",
]

N_STATES = 3
N_QUANTILES = 5

RANDOM_STATE = 42

# Number of non-overlapping observations.
# We use the largest horizon being tested.
NON_OVERLAP_STEP = 30

BOOTSTRAP_ITERATIONS = 2000


# ============================================================
# HELPERS
# ============================================================


def get_timestamp_series(df: pd.DataFrame) -> pd.Series:
    """
    Return the ET timestamp series from the dataframe.
    """

    if "timestamp ET" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp ET"],
            errors="coerce",
        )

        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("America/New_York")

        else:
            timestamps = timestamps.dt.tz_convert("America/New_York")

        return timestamps

    raise KeyError("Expected 'timestamp ET' column in DataFrame.")


def get_rth_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only RTH observations and prepare timestamp index.
    """

    df = df.copy()

    timestamps = get_timestamp_series(df)

    df["_timestamp_et"] = timestamps

    rth = df.loc[df["market_period"].eq("RTH")].copy()

    rth = rth.sort_values("_timestamp_et")

    rth = rth.set_index("_timestamp_et")

    rth.index.name = "timestamp_et"

    return rth


def calculate_train_quantile_bins(
    train: pd.DataFrame,
    feature: str,
) -> np.ndarray:
    """
    Calculate quantile boundaries exclusively from TRAIN data.
    """

    values = train[feature].replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        raise ValueError(f"No valid training observations for {feature}.")

    bins = values.quantile(
        np.linspace(
            0.0,
            1.0,
            N_QUANTILES + 1,
        )
    ).to_numpy()

    bins = np.unique(bins)

    if len(bins) < 3:
        raise ValueError(f"Unable to create sufficient quantile bins for {feature}.")

    return bins


def assign_quantiles(
    series: pd.Series,
    bins: np.ndarray,
) -> pd.Series:
    """
    Assign observations to train-defined quantile bins.
    """

    return pd.cut(
        series,
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )


def bootstrap_mean_difference(
    q1: np.ndarray,
    q5: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """
    Bootstrap the difference:

        mean(q5) - mean(q1)

    Returns:
        observed_difference,
        lower_95,
        upper_95
    """

    rng = np.random.default_rng(seed)

    q1 = np.asarray(q1, dtype=float)
    q5 = np.asarray(q5, dtype=float)

    q1 = q1[np.isfinite(q1)]
    q5 = q5[np.isfinite(q5)]

    if len(q1) == 0 or len(q5) == 0:
        return np.nan, np.nan, np.nan

    observed = q5.mean() - q1.mean()

    differences = np.empty(iterations)

    for i in range(iterations):
        sample_q1 = rng.choice(
            q1,
            size=len(q1),
            replace=True,
        )

        sample_q5 = rng.choice(
            q5,
            size=len(q5),
            replace=True,
        )

        differences[i] = sample_q5.mean() - sample_q1.mean()

    lower = np.percentile(
        differences,
        2.5,
    )

    upper = np.percentile(
        differences,
        97.5,
    )

    return observed, lower, upper


def permutation_test(
    q1: np.ndarray,
    q5: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> float:
    """
    Two-sided permutation test for the difference in means.

    H0:
        Q1 and Q5 have the same mean.

    Returns a two-sided p-value.
    """

    rng = np.random.default_rng(seed)

    q1 = np.asarray(q1, dtype=float)
    q5 = np.asarray(q5, dtype=float)

    q1 = q1[np.isfinite(q1)]
    q5 = q5[np.isfinite(q5)]

    if len(q1) == 0 or len(q5) == 0:
        return np.nan

    observed = q5.mean() - q1.mean()

    combined = np.concatenate([q1, q5])

    n_q1 = len(q1)

    count = 0

    for _ in range(iterations):
        shuffled = rng.permutation(combined)

        perm_q1 = shuffled[:n_q1]
        perm_q5 = shuffled[n_q1:]

        difference = perm_q5.mean() - perm_q1.mean()

        if abs(difference) >= abs(observed):
            count += 1

    return (count + 1) / (iterations + 1)


def calculate_profit_factor(
    returns: np.ndarray,
) -> float:
    """
    Profit factor:

        gross positive returns
        ----------------------
        gross negative returns
    """

    returns = np.asarray(
        returns,
        dtype=float,
    )

    returns = returns[np.isfinite(returns)]

    gains = returns[returns > 0].sum()

    losses = -returns[returns < 0].sum()

    if losses == 0:
        if gains > 0:
            return np.inf

        return np.nan

    return gains / losses


def calculate_win_rate(
    returns: np.ndarray,
) -> float:
    """
    Fraction of positive returns.
    """

    returns = np.asarray(
        returns,
        dtype=float,
    )

    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return np.nan

    return np.mean(returns > 0)


def print_effect(
    *,
    state: int,
    feature: str,
    target: str,
    q1: np.ndarray,
    q5: np.ndarray,
) -> None:
    """
    Print LONG and SHORT mean-reversion statistics.

    Mean-reversion interpretation:

        Q1 feature → LONG

        Q5 feature → SHORT

    Therefore:

        LONG edge  = Q1 mean return

        SHORT edge = -Q5 mean return
    """

    q1 = np.asarray(q1, dtype=float)
    q5 = np.asarray(q5, dtype=float)

    q1 = q1[np.isfinite(q1)]
    q5 = q5[np.isfinite(q5)]

    if len(q1) == 0 or len(q5) == 0:
        return

    long_returns = q1
    short_returns = -q5

    long_mean = long_returns.mean()
    short_mean = short_returns.mean()

    long_wr = calculate_win_rate(long_returns)

    short_wr = calculate_win_rate(short_returns)

    long_pf = calculate_profit_factor(long_returns)

    short_pf = calculate_profit_factor(short_returns)

    # For the directional mean-reversion comparison,
    # both sides are represented as positive expectancy:
    #
    # Q1 return  -> LONG
    # -Q5 return -> SHORT
    #
    # Difference between the raw groups is also reported.

    raw_difference = q1.mean() - q5.mean()

    print(f"\nSTATE {state}")

    print("\nLONG")
    print(f"Q1 observations: {len(q1)}")
    print(f"Q1 mean return: {long_mean:.10f}")
    print(f"Q1 win rate: {long_wr:.4%}")
    print(f"Q1 profit factor: {long_pf:.4f}")

    print("\nSHORT")
    print(f"Q5 observations: {len(q5)}")
    print(f"Q5 mean return: {short_mean:.10f}")
    print(f"Q5 win rate: {short_wr:.4%}")
    print(f"Q5 profit factor: {short_pf:.4f}")

    print(f"\nQ1 - Q5 raw return difference: {raw_difference:.10f}")

    # Bootstrap the mean-reversion edge.
    #
    # LONG edge:
    #   mean(Q1)
    #
    # SHORT edge:
    #   mean(-Q5)
    #
    # We therefore bootstrap the direct directional
    # expectancy for each side.

    rng = np.random.default_rng(RANDOM_STATE)

    long_bootstrap = np.empty(BOOTSTRAP_ITERATIONS)

    short_bootstrap = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):
        long_sample = rng.choice(
            long_returns,
            size=len(long_returns),
            replace=True,
        )

        short_sample = rng.choice(
            short_returns,
            size=len(short_returns),
            replace=True,
        )

        long_bootstrap[i] = long_sample.mean()

        short_bootstrap[i] = short_sample.mean()

    long_lower = np.percentile(
        long_bootstrap,
        2.5,
    )

    long_upper = np.percentile(
        long_bootstrap,
        97.5,
    )

    short_lower = np.percentile(
        short_bootstrap,
        2.5,
    )

    short_upper = np.percentile(
        short_bootstrap,
        97.5,
    )

    print(f"LONG 95% CI: [{long_lower:.10f}, {long_upper:.10f}]")

    print(f"SHORT 95% CI: [{short_lower:.10f}, {short_upper:.10f}]")

    # Direct permutation tests against zero.
    #
    # For the purposes of this benchmark, this tests
    # whether the directional return has a non-zero mean.

    def zero_mean_permutation(
        values: np.ndarray,
        iterations: int = BOOTSTRAP_ITERATIONS,
        seed: int = RANDOM_STATE,
    ) -> float:

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan

        observed = values.mean()

        rng_local = np.random.default_rng(seed)

        centered = values - observed

        count = 0

        for _ in range(iterations):
            signs = rng_local.choice(
                [-1.0, 1.0],
                size=len(centered),
            )

            statistic = np.mean(centered * signs)

            if abs(statistic) >= abs(observed):
                count += 1

        return (count + 1) / (iterations + 1)

    long_p = zero_mean_permutation(
        long_returns,
        seed=RANDOM_STATE,
    )

    short_p = zero_mean_permutation(
        short_returns,
        seed=RANDOM_STATE + 1,
    )

    print(f"LONG permutation p-value: {long_p:.4f}")

    print(f"SHORT permutation p-value: {short_p:.4f}")

    if long_lower > 0 and long_p < 0.05:
        print("LONG RESULT: POSITIVE MEAN-REVERSION EFFECT")
    else:
        print("LONG RESULT: NOT STATISTICALLY ROBUST")

    if short_lower > 0 and short_p < 0.05:
        print("SHORT RESULT: POSITIVE MEAN-REVERSION EFFECT")
    else:
        print("SHORT RESULT: NOT STATISTICALLY ROBUST")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)

    print("MEAN REVERSION — OOS VALIDATION")

    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    df = load_data()

    rth = get_rth_data(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # ---------------------------------------------------------
    # TRAIN / OOS SPLIT
    # ---------------------------------------------------------

    train = rth.loc[rth.index <= TRAIN_END].copy()

    oos = rth.loc[rth.index > TRAIN_END].copy()

    print("\n=== DATA SPLIT ===")

    print(f"Train observations: {len(train)}")

    print(f"Train start: {train.index.min()}")

    print(f"Train end: {train.index.max()}")

    print(f"OOS observations: {len(oos)}")

    print(f"OOS start: {oos.index.min()}")

    print(f"OOS end: {oos.index.max()}")

    if train.empty:
        raise ValueError("Training dataset is empty.")

    if oos.empty:
        raise ValueError("OOS dataset is empty.")

    # ---------------------------------------------------------
    # FIT HMM ONLY ON TRAIN
    # ---------------------------------------------------------

    print("\n=== FITTING HMM ON TRAIN ===")

    hmm = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    hmm.fit(train)

    train_states = hmm.predict_states(train)

    oos_states = hmm.predict_states(oos)

    train = train.copy()
    oos = oos.copy()

    train["hmm_state"] = train_states
    oos["hmm_state"] = oos_states

    print(f"Converged: {hmm.model.monitor_.converged}")

    print(f"Iterations: {hmm.model.monitor_.iter}")

    print("\nTrain state proportions:")

    print(train["hmm_state"].value_counts(normalize=True).sort_index())

    print("\nOOS state proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # ---------------------------------------------------------
    # FEATURE LOOP
    # ---------------------------------------------------------

    for feature in FEATURES:
        print("\n" + "#" * 70)

        print(f"FEATURE: {feature}")

        print("#" * 70)

        # -----------------------------------------------------
        # QUANTILES ARE FIT ONLY ON TRAIN
        # -----------------------------------------------------

        bins = calculate_train_quantile_bins(
            train,
            feature,
        )

        print("\nTrain quantile boundaries:")

        for i, boundary in enumerate(bins):
            print(f"Boundary {i}: {boundary:.10f}")

        oos = oos.copy()

        oos["mean_reversion_quantile"] = assign_quantiles(
            oos[feature],
            bins,
        )

        # -----------------------------------------------------
        # TARGET LOOP
        # -----------------------------------------------------

        for target in TARGETS:
            print("\n" + "-" * 70)

            print(f"{feature} -> {target}")

            print("-" * 70)

            required = [
                feature,
                target,
                "hmm_state",
                "mean_reversion_quantile",
            ]

            data = oos[required].dropna().copy()

            if data.empty:
                print("No valid OOS observations.")
                continue

            # -------------------------------------------------
            # NON-OVERLAPPING SAMPLE
            # -------------------------------------------------
            #
            # We deliberately sample every 30th observation.
            # This is conservative because 30 is the largest
            # forward horizon being evaluated.
            #
            # This reduces dependence between observations.

            data = data.iloc[::NON_OVERLAP_STEP].copy()

            print(f"Non-overlapping observations: {len(data)}")

            # -------------------------------------------------
            # STATE LOOP
            # -------------------------------------------------

            for state in range(N_STATES):
                state_data = data.loc[data["hmm_state"] == state]

                if state_data.empty:
                    continue

                q1_data = state_data.loc[
                    state_data["mean_reversion_quantile"] == 0,
                    target,
                ].dropna()

                q5_data = state_data.loc[
                    state_data["mean_reversion_quantile"] == N_QUANTILES - 1,
                    target,
                ].dropna()

                if len(q1_data) < 100 or len(q5_data) < 100:
                    print(f"\nSTATE {state}: insufficient observations.")
                    continue

                print_effect(
                    state=state,
                    feature=feature,
                    target=target,
                    q1=q1_data.to_numpy(),
                    q5=q5_data.to_numpy(),
                )

    print("\n" + "=" * 70)

    print("MEAN REVERSION VALIDATION COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
