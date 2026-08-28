from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_YEARS = 2
VALIDATION_MONTHS = 3

# We test several absolute scales so RR is not confused
# with the size of the trade.
#
# reward:risk families:
#
# 2.00 : 1.00
# 1.50 : 1.00
# 1.00 : 1.00
# 0.75 : 1.00
# 0.67 : 1.00
# 0.50 : 1.00

PAYOFF_FAMILIES = {
    "2.00R": (2.00, 1.00),
    "1.50R": (1.50, 1.00),
    "1.00R": (1.00, 1.00),
    "0.75R": (0.75, 1.00),
    "0.67R": (0.67, 1.00),
    "0.50R": (0.50, 1.00),
}

# Absolute risk scales in NQ points.
RISK_SCALES = [5, 10, 15, 20]

# Short holding periods are the primary research area.
HORIZONS = [5, 10, 15, 30]


# ============================================================
# TIMESTAMP
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


# ============================================================
# PREPARE RTH
# ============================================================


def prepare_rth(
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

    if "session_date" in rth.columns:
        rth["_session_id"] = rth["session_date"].astype(str)
    else:
        rth["_session_id"] = rth.index.date

    return rth


# ============================================================
# FIRST BARRIER
# ============================================================


def first_barrier_outcome(
    session: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:
    """
    +1 = target reached first
    -1 = stop reached first
     0 = unresolved / ambiguous

    IMPORTANT:
    Target and stop are evaluated only inside the same
    RTH session.

    If both barriers are touched in the same candle,
    the ordering cannot be determined from OHLC data.
    Therefore the observation is treated as unresolved.
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
# SESSION-AWARE BARRIER ENGINE
# ============================================================


def calculate_outcomes(
    df: pd.DataFrame,
    target_points: float,
    stop_points: float,
    horizon: int,
    direction: str,
) -> pd.Series:

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
            session=session,
            target_points=target_points,
            stop_points=stop_points,
            horizon=horizon,
            direction=direction,
        )

        result.loc[session_result.index] = session_result

    return result


# ============================================================
# TRADE STATISTICS
# ============================================================


def calculate_stats(
    outcomes: pd.Series,
    target_points: float,
    stop_points: float,
) -> dict[str, float]:

    wins = int((outcomes == 1).sum())

    losses = int((outcomes == -1).sum())

    unresolved = int((outcomes == 0).sum())

    resolved = wins + losses

    total = len(outcomes)

    if resolved == 0:
        return {
            "wins": wins,
            "losses": losses,
            "unresolved": unresolved,
            "resolution_rate": 0.0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
        }

    win_rate = wins / resolved

    gross_profit = wins * target_points

    gross_loss = losses * stop_points

    expectancy = (gross_profit - gross_loss) / resolved

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    return {
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolution_rate": (resolved / total),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    rth: pd.DataFrame,
) -> list[
    tuple[
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:

    start = rth.index.min()
    end = rth.index.max()

    validation_start = start + pd.DateOffset(years=TRAIN_YEARS)

    windows = []

    while validation_start < end:
        train_start = validation_start - pd.DateOffset(years=TRAIN_YEARS)

        validation_end = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)

        validation_end = min(validation_end, end)

        windows.append(
            (
                train_start,
                validation_start,
                validation_end,
            )
        )

        validation_start = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)

    return windows


# ============================================================
# EVALUATE ONE CONFIGURATION
# ============================================================


def evaluate_configuration(
    df: pd.DataFrame,
    direction: str,
    horizon: int,
    target_points: float,
    stop_points: float,
) -> dict[str, float]:

    outcomes = calculate_outcomes(
        df=df,
        target_points=target_points,
        stop_points=stop_points,
        horizon=horizon,
        direction=direction,
    )

    stats = calculate_stats(
        outcomes=outcomes,
        target_points=target_points,
        stop_points=stop_points,
    )

    return stats


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================


def run_walk_forward(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    windows = generate_windows(rth)

    rows = []

    print(f"\nWalk-forward windows: {len(windows)}")

    for window_number, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        validation = rth.loc[
            (rth.index >= validation_start) & (rth.index < validation_end)
        ].copy()

        if validation.empty:
            continue

        print(
            f"\nWindow {window_number}: "
            f"{validation_start.date()} → "
            f"{validation_end.date()}"
        )

        for horizon in HORIZONS:
            for direction in [
                "long",
                "short",
            ]:
                for family_name, (
                    reward_multiple,
                    risk_multiple,
                ) in PAYOFF_FAMILIES.items():
                    for risk_points in RISK_SCALES:
                        target_points = risk_points * reward_multiple

                        stop_points = risk_points * risk_multiple

                        stats = evaluate_configuration(
                            df=validation,
                            direction=direction,
                            horizon=horizon,
                            target_points=target_points,
                            stop_points=stop_points,
                        )

                        rows.append(
                            {
                                "window": window_number,
                                "validation_start": validation_start,
                                "validation_end": validation_end,
                                "direction": direction,
                                "horizon": horizon,
                                "payoff_family": family_name,
                                "reward_multiple": reward_multiple,
                                "risk_multiple": risk_multiple,
                                "risk_points": risk_points,
                                "target_points": target_points,
                                "stop_points": stop_points,
                                **stats,
                            }
                        )

    return pd.DataFrame(rows)


# ============================================================
# STABILITY SUMMARY
# ============================================================


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        results.groupby(
            [
                "direction",
                "horizon",
                "payoff_family",
            ]
        )
        .agg(
            windows=(
                "window",
                "nunique",
            ),
            positive_windows=(
                "expectancy",
                lambda x: (x > 0).sum(),
            ),
            mean_expectancy=(
                "expectancy",
                "mean",
            ),
            median_expectancy=(
                "expectancy",
                "median",
            ),
            mean_win_rate=(
                "win_rate",
                "mean",
            ),
            mean_profit_factor=(
                "profit_factor",
                "mean",
            ),
            mean_resolution=(
                "resolution_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["positive_window_rate"] = summary["positive_windows"] / summary["windows"]

    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================


def print_summary(
    summary: pd.DataFrame,
) -> None:

    print("\n" + "=" * 100)

    print("PAYOFF GEOMETRY STABILITY SUMMARY")

    print("=" * 100)

    display = summary.copy()

    display["positive_window_rate"] = display["positive_window_rate"].map(
        lambda x: f"{x:.1%}"
    )

    display["mean_expectancy"] = display["mean_expectancy"].map(lambda x: f"{x:.4f}")

    display["median_expectancy"] = display["median_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    display["mean_win_rate"] = display["mean_win_rate"].map(lambda x: f"{x:.2%}")

    display["mean_profit_factor"] = display["mean_profit_factor"].map(
        lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
    )

    display["mean_resolution"] = display["mean_resolution"].map(lambda x: f"{x:.2%}")

    print(display.to_string(index=False))


# ============================================================
# ROBUSTNESS FILTER
# ============================================================


def print_robust_candidates(
    summary: pd.DataFrame,
) -> None:
    """
    Diagnostic filter only.

    This DOES NOT select the final strategy.

    A candidate is considered interesting if:

        positive window rate >= 75%
        median expectancy > 0
        mean resolution >= 50%
    """

    candidates = summary.loc[
        (summary["positive_window_rate"] >= 0.75)
        & (summary["median_expectancy"] > 0)
        & (summary["mean_resolution"] >= 0.50)
    ].copy()

    print("\n" + "=" * 100)

    print("ROBUST CANDIDATE PAYOFF FAMILIES")

    print("=" * 100)

    if candidates.empty:
        print("No payoff family passed the diagnostic filter.")

        print("This is NOT a failure.")

        print("It means the barrier geometry needs further research.")

        return

    candidates = candidates.sort_values(
        [
            "direction",
            "horizon",
            "median_expectancy",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    display = candidates[
        [
            "direction",
            "horizon",
            "payoff_family",
            "windows",
            "positive_window_rate",
            "median_expectancy",
            "mean_expectancy",
            "mean_win_rate",
            "mean_profit_factor",
            "mean_resolution",
        ]
    ].copy()

    display["positive_window_rate"] = display["positive_window_rate"].map(
        lambda x: f"{x:.1%}"
    )

    display["median_expectancy"] = display["median_expectancy"].map(
        lambda x: f"{x:.4f}"
    )

    display["mean_expectancy"] = display["mean_expectancy"].map(lambda x: f"{x:.4f}")

    display["mean_win_rate"] = display["mean_win_rate"].map(lambda x: f"{x:.2%}")

    display["mean_profit_factor"] = display["mean_profit_factor"].map(
        lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
    )

    display["mean_resolution"] = display["mean_resolution"].map(lambda x: f"{x:.2%}")

    print(display.to_string(index=False))


# ============================================================
# SCALE STABILITY
# ============================================================


def print_scale_stability(
    results: pd.DataFrame,
) -> None:
    """
    Checks whether a payoff family survives across
    different absolute point scales.

    This matters because we do not want an apparent edge
    that only exists at one arbitrary stop distance.
    """

    scale_summary = (
        results.groupby(
            [
                "direction",
                "horizon",
                "payoff_family",
                "risk_points",
            ]
        )
        .agg(
            mean_expectancy=(
                "expectancy",
                "mean",
            ),
            positive_windows=(
                "expectancy",
                lambda x: (x > 0).mean(),
            ),
            mean_win_rate=(
                "win_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\n" + "=" * 100)

    print("ABSOLUTE SCALE STABILITY")

    print("=" * 100)

    display = scale_summary.copy()

    display["mean_expectancy"] = display["mean_expectancy"].map(lambda x: f"{x:.4f}")

    display["positive_windows"] = display["positive_windows"].map(lambda x: f"{x:.1%}")

    display["mean_win_rate"] = display["mean_win_rate"].map(lambda x: f"{x:.2%}")

    print(display.to_string(index=False))


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)

    print("PAYOFF GEOMETRY — CONTROLLED WALK-FORWARD VALIDATION")

    print("=" * 100)

    print("\nThis experiment does NOT select a final TP/SL.")

    print("It compares predefined payoff families across unseen")

    print("walk-forward validation periods.")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    rth = prepare_rth(df)

    print(f"\nRTH observations: {len(rth)}")

    print(f"RTH sessions: {rth['_session_id'].nunique()}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    results = run_walk_forward(rth)

    if results.empty:
        print("\nNo results generated.")

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = build_summary(results)

    print_summary(summary)

    # --------------------------------------------------------
    # ROBUST CANDIDATES
    # --------------------------------------------------------

    print_robust_candidates(summary)

    # --------------------------------------------------------
    # SCALE STABILITY
    # --------------------------------------------------------

    print_scale_stability(results)

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("PAYOFF GEOMETRY VALIDATION COMPLETE")

    print("=" * 100)

    print("\nNo final payoff configuration has been selected.")

    print("The next decision will be based on robustness,")

    print("not maximum historical expectancy.")

    print("\nIMPORTANT:")

    print("No transaction costs.")

    print("No slippage.")

    print("No position sizing.")

    print("No funded-account constraints.")

    print("No ML model.")

    print("Those are deliberately postponed until the payoff")

    print("geometry survives this research stage.")


if __name__ == "__main__":
    main()
