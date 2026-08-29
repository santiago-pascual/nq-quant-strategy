from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_END = pd.Timestamp(
    "2024-12-31 23:59:00",
    tz="America/New_York",
)

RANDOM_STATE = 42
N_STATES = 3

# Horizons in bars.
# We deliberately examine several horizons rather than
# choosing one in advance.
HORIZONS = [5, 10, 15, 30]

# Price excursion levels in NQ points.
# These are measurement levels, NOT proposed strategy stops
# or targets.
LEVELS = [5, 10, 15, 20, 30, 40, 50]


# ============================================================
# DATA PREPARATION
# ============================================================


def get_timestamp_series(
    df: pd.DataFrame,
) -> pd.Series:

    if "timestamp ET" not in df.columns:
        raise KeyError("Expected 'timestamp ET' column.")

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    return timestamps


def prepare_rth_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_timestamp_et"] = get_timestamp_series(df)

    if "market_period" not in df.columns:
        raise KeyError("Missing market_period column.")

    rth = df.loc[df["market_period"] == "RTH"].copy()

    rth = rth.sort_values("_timestamp_et")

    rth = rth.set_index("_timestamp_et")

    rth.index.name = "timestamp_et"

    return rth


# ============================================================
# MFE / MAE CALCULATION
# ============================================================


def calculate_excursions(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Calculate forward Maximum Favorable Excursion (MFE)
    and Maximum Adverse Excursion (MAE).

    LONG:

        MFE = max(high[t+1:t+h]) - close[t]

        MAE = close[t] - min(low[t+1:t+h])

    SHORT:

        MFE = close[t] - min(low[t+1:t+h])

        MAE = max(high[t+1:t+h]) - close[t]

    All values are expressed in NQ points.

    IMPORTANT:
    These are descriptive forward measurements only.
    They are NOT used as features for the predictive model.
    """

    result = pd.DataFrame(index=df.index)

    close = df["close"].to_numpy(dtype=float)

    high = df["high"].to_numpy(dtype=float)

    low = df["low"].to_numpy(dtype=float)

    n = len(df)

    for horizon in horizons:
        long_mfe = np.full(
            n,
            np.nan,
        )

        long_mae = np.full(
            n,
            np.nan,
        )

        short_mfe = np.full(
            n,
            np.nan,
        )

        short_mae = np.full(
            n,
            np.nan,
        )

        for i in range(n):
            end = i + horizon + 1

            if end > n:
                continue

            entry = close[i]

            future_high = high[i + 1 : end]

            future_low = low[i + 1 : end]

            if len(future_high) == 0 or len(future_low) == 0:
                continue

            # -----------------------------
            # LONG
            # -----------------------------

            long_mfe[i] = np.max(future_high) - entry

            long_mae[i] = entry - np.min(future_low)

            # -----------------------------
            # SHORT
            # -----------------------------

            short_mfe[i] = entry - np.min(future_low)

            short_mae[i] = np.max(future_high) - entry

        result[f"long_mfe_{horizon}"] = long_mfe

        result[f"long_mae_{horizon}"] = long_mae

        result[f"short_mfe_{horizon}"] = short_mfe

        result[f"short_mae_{horizon}"] = short_mae

    return result


# ============================================================
# BARRIER ANALYSIS
# ============================================================


def calculate_barrier_statistics(
    mfe: pd.Series,
    mae: pd.Series,
    level: float,
) -> dict[str, float]:
    """
    Calculate simple excursion statistics.

    We intentionally do NOT claim that a target is reached
    before a stop here.

    This function only asks:

        How often did MFE reach X points?

        How often did MAE reach X points?

    The sequencing question will be handled later.
    """

    mfe = mfe.dropna()
    mae = mae.dropna()

    if len(mfe) == 0:
        return {
            "observations": 0,
            "mfe_hit_rate": np.nan,
            "mae_hit_rate": np.nan,
            "mfe_median": np.nan,
            "mae_median": np.nan,
        }

    return {
        "observations": len(mfe),
        "mfe_hit_rate": (mfe >= level).mean(),
        "mae_hit_rate": (mae >= level).mean(),
        "mfe_median": mfe.median(),
        "mae_median": mae.median(),
    }


# ============================================================
# SEQUENTIAL BARRIER ANALYSIS
# ============================================================


def calculate_first_barrier_outcome(
    df: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:
    """
    Determine which barrier is touched FIRST.

    Returns:

        +1 = target hit first
        -1 = stop hit first
         0 = neither hit within horizon

    LONG:

        target = entry + target_points
        stop   = entry - stop_points

    SHORT:

        target = entry - target_points
        stop   = entry + stop_points

    IMPORTANT:

    If both target and stop are inside the SAME OHLC bar,
    we cannot know which occurred first from OHLC data.

    Such observations are classified as 0 rather than
    inventing intrabar sequencing.
    """

    close = df["close"].to_numpy(dtype=float)

    high = df["high"].to_numpy(dtype=float)

    low = df["low"].to_numpy(dtype=float)

    n = len(df)

    outcomes = np.zeros(
        n,
        dtype=np.int8,
    )

    for i in range(n):
        end = i + horizon + 1

        if end > n:
            continue

        entry = close[i]

        future_high = high[i + 1 : end]

        future_low = low[i + 1 : end]

        if direction == "long":
            target_price = entry + target_points

            stop_price = entry - stop_points

            for bar_high, bar_low in zip(
                future_high,
                future_low,
            ):
                target_hit = bar_high >= target_price

                stop_hit = bar_low <= stop_price

                if target_hit and stop_hit:
                    # Unknown intrabar order.
                    outcomes[i] = 0
                    break

                if target_hit:
                    outcomes[i] = 1
                    break

                if stop_hit:
                    outcomes[i] = -1
                    break

        elif direction == "short":
            target_price = entry - target_points

            stop_price = entry + stop_points

            for bar_high, bar_low in zip(
                future_high,
                future_low,
            ):
                target_hit = bar_low <= target_price

                stop_hit = bar_high >= stop_price

                if target_hit and stop_hit:
                    # Unknown intrabar order.
                    outcomes[i] = 0
                    break

                if target_hit:
                    outcomes[i] = 1
                    break

                if stop_hit:
                    outcomes[i] = -1
                    break

        else:
            raise ValueError("direction must be 'long' or 'short'.")

    return pd.Series(
        outcomes,
        index=df.index,
    )


# ============================================================
# HMM REGIME
# ============================================================


def fit_hmm_regimes(
    train: pd.DataFrame,
    oos: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    VolatilityRegimeModel,
]:
    """
    Fit HMM only on training data and apply it to OOS.
    """

    hmm = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    hmm.fit(train)

    train = train.copy()
    oos = oos.copy()

    train["hmm_state"] = hmm.predict_states(train)

    oos["hmm_state"] = hmm.predict_states(oos)

    return (
        train,
        oos,
        hmm,
    )


# ============================================================
# PRINT EXCURSION SUMMARY
# ============================================================


def print_excursion_summary(
    df: pd.DataFrame,
    label: str,
) -> None:

    print("\n" + "=" * 70)

    print(f"{label} — EXCURSION DISTRIBUTIONS")

    print("=" * 70)

    for horizon in HORIZONS:
        print(f"\n--- {horizon}-BAR HORIZON ---")

        columns = [
            f"long_mfe_{horizon}",
            f"long_mae_{horizon}",
            f"short_mfe_{horizon}",
            f"short_mae_{horizon}",
        ]

        summary = (
            df[columns]
            .describe(
                percentiles=[
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                ]
            )
            .T
        )

        print(
            summary[
                [
                    "count",
                    "mean",
                    "25%",
                    "50%",
                    "75%",
                    "90%",
                    "95%",
                ]
            ]
        )


# ============================================================
# PRINT LEVEL ANALYSIS
# ============================================================


def print_level_analysis(
    df: pd.DataFrame,
    label: str,
) -> None:

    print("\n" + "=" * 70)

    print(f"{label} — MFE / MAE LEVEL ANALYSIS")

    print("=" * 70)

    for horizon in HORIZONS:
        print(f"\n--- {horizon}-BAR HORIZON ---")

        for level in LEVELS:
            long_stats = calculate_barrier_statistics(
                df[f"long_mfe_{horizon}"],
                df[f"long_mae_{horizon}"],
                level,
            )

            short_stats = calculate_barrier_statistics(
                df[f"short_mfe_{horizon}"],
                df[f"short_mae_{horizon}"],
                level,
            )

            print(f"\nLevel: {level} points")

            print(
                "LONG  "
                f"MFE hit: "
                f"{long_stats['mfe_hit_rate']:.2%} | "
                f"MAE hit: "
                f"{long_stats['mae_hit_rate']:.2%}"
            )

            print(
                "SHORT "
                f"MFE hit: "
                f"{short_stats['mfe_hit_rate']:.2%} | "
                f"MAE hit: "
                f"{short_stats['mae_hit_rate']:.2%}"
            )


# ============================================================
# REGIME ANALYSIS
# ============================================================


def print_regime_analysis(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)

    print("OOS MFE / MAE BY HMM REGIME")

    print("=" * 70)

    for horizon in HORIZONS:
        print(f"\n--- {horizon}-BAR HORIZON ---")

        columns = [
            f"long_mfe_{horizon}",
            f"long_mae_{horizon}",
            f"short_mfe_{horizon}",
            f"short_mae_{horizon}",
        ]

        regime_summary = df.groupby("hmm_state")[columns].agg(
            [
                "mean",
                "median",
            ]
        )

        print(regime_summary)


# ============================================================
# BARRIER GRID
# ============================================================


def print_barrier_grid(
    df: pd.DataFrame,
    direction: str,
) -> None:
    """
    Evaluate several target/stop combinations.

    This is still descriptive.

    We are NOT selecting the best combination.

    The purpose is to see the general geometry of the
    opportunity space.
    """

    print("\n" + "=" * 70)

    print(f"{direction.upper()} — TARGET / STOP GRID")

    print("=" * 70)

    target_levels = [
        5,
        10,
        15,
        20,
    ]

    stop_levels = [
        5,
        10,
        15,
        20,
    ]

    horizon = 15

    for target in target_levels:
        for stop in stop_levels:
            outcomes = calculate_first_barrier_outcome(
                df,
                target_points=target,
                stop_points=stop,
                horizon=horizon,
                direction=direction,
            )

            valid = outcomes[outcomes != 0]

            if len(valid) == 0:
                continue

            wins = (valid == 1).sum()

            losses = (valid == -1).sum()

            total = len(valid)

            win_rate = wins / total

            expectancy = ((wins * target) - (losses * stop)) / total

            print(
                f"Target {target:>2} / "
                f"Stop {stop:>2} | "
                f"Resolved {total:>6} | "
                f"Win rate {win_rate:>7.2%} | "
                f"Raw expectancy {expectancy:>9.3f} pts"
            )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)

    print("MFE / MAE TRADE-OUTCOME ANALYSIS")

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth_data(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # --------------------------------------------------------
    # TRAIN / OOS SPLIT
    # --------------------------------------------------------

    train = rth.loc[rth.index <= TRAIN_END].copy()

    oos = rth.loc[rth.index > TRAIN_END].copy()

    print("\n=== DATA SPLIT ===")

    print(f"Train observations: {len(train)}")

    print(f"Train start: {train.index.min()}")

    print(f"Train end: {train.index.max()}")

    print(f"OOS observations: {len(oos)}")

    print(f"OOS start: {oos.index.min()}")

    print(f"OOS end: {oos.index.max()}")

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    print("\n=== FITTING HMM ON TRAIN ===")

    (
        train,
        oos,
        hmm,
    ) = fit_hmm_regimes(
        train,
        oos,
    )

    print(f"Converged: {hmm.model.monitor_.converged}")

    print(f"Iterations: {hmm.model.monitor_.iter}")

    print("\nOOS regime proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # --------------------------------------------------------
    # EXCURSIONS
    # --------------------------------------------------------

    print("\n=== CALCULATING OOS EXCURSIONS ===")

    excursions = calculate_excursions(
        oos,
        HORIZONS,
    )

    oos = oos.join(excursions)

    # --------------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------------

    print_excursion_summary(
        oos,
        "OOS",
    )

    # --------------------------------------------------------
    # LEVEL ANALYSIS
    # --------------------------------------------------------

    print_level_analysis(
        oos,
        "OOS",
    )

    # --------------------------------------------------------
    # REGIME ANALYSIS
    # --------------------------------------------------------

    print_regime_analysis(oos)

    # --------------------------------------------------------
    # BARRIER GRID
    # --------------------------------------------------------

    print_barrier_grid(
        oos,
        "long",
    )

    print_barrier_grid(
        oos,
        "short",
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("MFE / MAE ANALYSIS COMPLETE")

    print("=" * 70)

    print("\nIMPORTANT:")

    print("The target/stop grid is descriptive only.")

    print("No trading rule has been selected.")

    print("No parameter optimization has been performed.")

    print("The next step is to use the observed excursion")

    print("structure to construct statistically defensible")

    print("LONG and SHORT classification targets.")


if __name__ == "__main__":
    main()
