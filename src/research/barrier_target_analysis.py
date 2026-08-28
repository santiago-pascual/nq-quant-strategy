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

# Holding periods in 1-minute RTH bars.
HORIZONS = [5, 10, 15, 30]

# Candidate target / stop distances in NQ points.
TARGETS = [5, 10, 15, 20]
STOPS = [5, 10, 15, 20]


# ============================================================
# TIMESTAMP / RTH PREPARATION
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
# SESSION IDENTIFICATION
# ============================================================


def add_session_identifier(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "session_date" in df.columns:
        df["_session_id"] = df["session_date"].astype(str)
    else:
        df["_session_id"] = df.index.date

    return df


# ============================================================
# HMM
# ============================================================


def fit_hmm(
    train: pd.DataFrame,
    oos: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    VolatilityRegimeModel,
]:

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
# FIRST BARRIER ENGINE
# ============================================================


def first_barrier_outcome(
    session: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:
    """
    Determine which barrier is reached first.

    Returns:

        +1 = target reached first
        -1 = stop reached first
         0 = neither reached
         0 = ambiguous same-bar target/stop

    IMPORTANT:

    Only bars inside the same RTH session are examined.

    If target and stop are both touched inside the same
    1-minute candle, the ordering is unknowable from OHLC.
    That observation is therefore treated as unresolved.
    """

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    n = len(session)

    outcomes = np.zeros(
        n,
        dtype=np.int8,
    )

    for i in range(n):
        end = min(
            i + horizon + 1,
            n,
        )

        if i + 1 >= end:
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
        index=session.index,
    )


# ============================================================
# APPLY BARRIER ENGINE SESSION BY SESSION
# ============================================================


def calculate_barrier_outcomes(
    df: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:
    """
    Apply first_barrier_outcome independently to each RTH
    session so no trade can cross into another trading day.
    """

    result = pd.Series(
        0,
        index=df.index,
        dtype="int8",
    )

    for _, session in df.groupby(
        "_session_id",
        sort=False,
    ):
        session_result = first_barrier_outcome(
            session,
            target_points,
            stop_points,
            horizon,
            direction,
        )

        result.loc[session_result.index] = session_result

    return result


# ============================================================
# STATISTICS
# ============================================================


def calculate_statistics(
    outcomes: pd.Series,
    target_points: float,
    stop_points: float,
) -> dict[str, float]:

    total = len(outcomes)

    wins = int((outcomes == 1).sum())

    losses = int((outcomes == -1).sum())

    unresolved = int((outcomes == 0).sum())

    resolved = wins + losses

    if resolved > 0:
        win_rate = wins / resolved

        raw_expectancy = ((wins * target_points) - (losses * stop_points)) / resolved

    else:
        win_rate = np.nan
        raw_expectancy = np.nan

    return {
        "observations": total,
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolved": resolved,
        "resolution_rate": (resolved / total if total > 0 else np.nan),
        "win_rate": win_rate,
        "raw_expectancy_points": (raw_expectancy),
    }


# ============================================================
# GRID ANALYSIS
# ============================================================


def run_barrier_grid(
    df: pd.DataFrame,
    direction: str,
    horizon: int,
) -> pd.DataFrame:

    rows = []

    for target in TARGETS:
        for stop in STOPS:
            outcomes = calculate_barrier_outcomes(
                df,
                target_points=target,
                stop_points=stop,
                horizon=horizon,
                direction=direction,
            )

            stats = calculate_statistics(
                outcomes,
                target,
                stop,
            )

            rows.append(
                {
                    "direction": direction,
                    "horizon": horizon,
                    "target": target,
                    "stop": stop,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# REGIME ANALYSIS
# ============================================================


def run_regime_analysis(
    df: pd.DataFrame,
    direction: str,
    horizon: int,
    target: float,
    stop: float,
) -> pd.DataFrame:
    """
    Calculate barrier statistics separately by HMM state.
    """

    rows = []

    for state, regime_df in df.groupby(
        "hmm_state",
        sort=True,
    ):
        outcomes = calculate_barrier_outcomes(
            regime_df,
            target_points=target,
            stop_points=stop,
            horizon=horizon,
            direction=direction,
        )

        stats = calculate_statistics(
            outcomes,
            target,
            stop,
        )

        rows.append(
            {
                "hmm_state": state,
                **stats,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRINT GRID
# ============================================================


def print_grid(
    grid: pd.DataFrame,
) -> None:

    display_columns = [
        "target",
        "stop",
        "wins",
        "losses",
        "unresolved",
        "resolution_rate",
        "win_rate",
        "raw_expectancy_points",
    ]

    printable = grid[display_columns].copy()

    printable["resolution_rate"] = printable["resolution_rate"].map(
        lambda x: f"{x:.2%}"
    )

    printable["win_rate"] = printable["win_rate"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else "nan"
    )

    printable["raw_expectancy_points"] = printable["raw_expectancy_points"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "nan"
    )

    print(printable.to_string(index=False))


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)

    print("SESSION-AWARE BARRIER TARGET ANALYSIS")

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth_data(df)

    rth = add_session_identifier(rth)

    print(f"\nRTH observations: {len(rth)}")

    print(f"RTH sessions: {rth['_session_id'].nunique()}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # --------------------------------------------------------
    # TRAIN / OOS
    # --------------------------------------------------------

    train = rth.loc[rth.index <= TRAIN_END].copy()

    oos = rth.loc[rth.index > TRAIN_END].copy()

    print("\n=== DATA SPLIT ===")

    print(f"Train: {len(train)}")

    print(f"OOS: {len(oos)}")

    print(f"Train end: {train.index.max()}")

    print(f"OOS start: {oos.index.min()}")

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    print("\n=== FITTING HMM ON TRAIN ===")

    (
        train,
        oos,
        hmm,
    ) = fit_hmm(
        train,
        oos,
    )

    print(f"Converged: {hmm.model.monitor_.converged}")

    print(f"Iterations: {hmm.model.monitor_.iter}")

    print("\nOOS regime proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # --------------------------------------------------------
    # BARRIER GRIDS
    # --------------------------------------------------------

    for horizon in HORIZONS:
        print("\n" + "=" * 70)

        print(f"OOS HORIZON: {horizon} BARS")

        print("=" * 70)

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        print("\n" + "-" * 70)

        print("LONG")

        print("-" * 70)

        long_grid = run_barrier_grid(
            oos,
            "long",
            horizon,
        )

        print_grid(long_grid)

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        print("\n" + "-" * 70)

        print("SHORT")

        print("-" * 70)

        short_grid = run_barrier_grid(
            oos,
            "short",
            horizon,
        )

        print_grid(short_grid)

    # --------------------------------------------------------
    # REGIME ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("REGIME-CONDITIONAL BARRIER ANALYSIS")

    print("=" * 70)

    # We use 15 bars for the first regime comparison
    # because it was the most informative horizon in the
    # previous directional research.

    regime_horizon = 15

    # These are diagnostic combinations, NOT selected
    # trading parameters.

    diagnostic_barriers = [
        (5, 10),
        (5, 15),
        (10, 15),
        (10, 20),
        (15, 20),
    ]

    for direction in [
        "long",
        "short",
    ]:
        print("\n" + "-" * 70)

        print(f"{direction.upper()} — {regime_horizon}-BAR REGIME ANALYSIS")

        print("-" * 70)

        for target, stop in diagnostic_barriers:
            print(f"\nTarget {target} / Stop {stop}")

            regime_results = run_regime_analysis(
                oos,
                direction,
                regime_horizon,
                target,
                stop,
            )

            printable = regime_results[
                [
                    "hmm_state",
                    "wins",
                    "losses",
                    "unresolved",
                    "resolution_rate",
                    "win_rate",
                    "raw_expectancy_points",
                ]
            ].copy()

            printable["resolution_rate"] = printable["resolution_rate"].map(
                lambda x: f"{x:.2%}"
            )

            printable["win_rate"] = printable["win_rate"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "nan"
            )

            printable["raw_expectancy_points"] = printable["raw_expectancy_points"].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "nan"
            )

            print(printable.to_string(index=False))

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("SESSION-AWARE BARRIER ANALYSIS COMPLETE")

    print("=" * 70)

    print("\nNo target/stop combination has been selected.")

    print("The results are descriptive research only.")

    print("The next step is to identify barrier definitions")

    print("that are sufficiently frequent, asymmetric, and")

    print("stable across OOS regimes before creating ML labels.")


if __name__ == "__main__":
    main()
