from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features

from src.research.s2_regime_router import (
    CANDIDATES,
    RANDOM_STATE,
    TARGET_STATE,
    TAIL_PERCENT,
    HORIZON,
    STOP_POINTS,
    BASE_FEATURES,
    VOL_BUCKETS,
    generate_windows,
    prepare_rth,
    calculate_thresholds,
    calculate_quality_scales,
    add_volatility_percentile,
    generate_candidate_trades,
    calculate_metrics,
    volatility_bucket,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# S2 RELATIVE REGIME ROUTER
# ============================================================
#
# PURPOSE
# -------
# Test whether the two already-discovered frozen strategies
# can be selected conditionally by volatility regime.
#
#
# FROZEN CANDIDATES
# -----------------
#
# A:
#   Quality >= 0.65
#   RR = 1.25
#
# B:
#   Quality >= 0.75
#   RR = 1.30
#
# Shared:
#   HMM State 2
#   17.5% lower-tail conditions
#   20 point stop
#   15 bar horizon
#
#
# IMPORTANT
# ---------
#
# NO optimization of A or B.
#
# NO XGBoost.
#
# NO OOS information is used to select the model.
#
# The only thing being tested is the ROUTING.
#
#
# ROUTING LOGIC
# -------------
#
# For each volatility regime:
#
#   1. Measure A and B on TRAIN.
#   2. Require enough training trades.
#   3. Compare their training expectancy.
#   4. Select the better candidate if the better one
#      is sufficiently better than the other.
#   5. Otherwise OFF.
#
# The routing decision is frozen before OOS begins.
#
# ============================================================


# ============================================================
# ROUTER PARAMETERS
# ============================================================

MIN_TRAIN_TRADES = 30

# Minimum relative improvement in mean R required
# for one candidate to beat the other.
#
# Example:
#
# A = +0.04R
# B = +0.06R
#
# Relative improvement:
#
#   (0.06 - 0.04) / abs(0.04)
#   = 50%
#
# B can therefore be selected.
#
# If the difference is too small, OFF.
#
MIN_RELATIVE_EDGE = 0.10


# Minimum absolute expectancy required for
# the selected candidate.
#
# This prevents selecting a candidate merely because
# it is "less bad" than the other.
#
# IMPORTANT:
# This is deliberately small because we do not want
# the previous router's extremely restrictive
# mean_R > 0 + PF > 1 logic.
#
MIN_EXPECTANCY = 0.00


# ============================================================
# HMM
# ============================================================


def fit_hmm(train, validation):

    model = VolatilityRegimeModel(
        n_states=3,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    return train, validation


# ============================================================
# ASSIGN VOLATILITY TO TRADES
# ============================================================


def assign_trade_volatility(
    trades,
    market_data,
):

    trades = trades.copy()

    if trades.empty:
        return trades

    volatility = market_data[["vol_percentile"]].copy()

    trades["vol_percentile"] = (
        volatility["vol_percentile"]
        .reindex(
            trades["entry_timestamp"],
            method="ffill",
        )
        .to_numpy()
    )

    trades["vol_bucket"] = trades["vol_percentile"].apply(volatility_bucket)

    return trades


# ============================================================
# RELATIVE ROUTER
# ============================================================


def select_relative_models(
    train_trades,
):

    decisions = []

    for (
        low,
        high,
        bucket,
    ) in VOL_BUCKETS:
        bucket_results = {}

        for candidate in CANDIDATES:
            candidate_trades = train_trades.loc[
                (train_trades["candidate"] == candidate)
                & (train_trades["vol_percentile"] >= low)
                & (train_trades["vol_percentile"] < high)
            ]

            m = calculate_metrics(candidate_trades)

            bucket_results[candidate] = m

        A = bucket_results["A"]
        B = bucket_results["B"]

        selected = "OFF"

        # ----------------------------------------------------
        # Need enough evidence for BOTH candidates.
        # ----------------------------------------------------

        enough_A = A["trades"] >= MIN_TRAIN_TRADES

        enough_B = B["trades"] >= MIN_TRAIN_TRADES

        if enough_A and enough_B:
            A_mean = A["mean_R"]
            B_mean = B["mean_R"]

            # ------------------------------------------------
            # Both positive:
            #
            # Select whichever has higher expectancy,
            # but only if the difference is meaningful.
            # ------------------------------------------------

            if A_mean > MIN_EXPECTANCY and B_mean > MIN_EXPECTANCY:
                if A_mean > B_mean:
                    relative_difference = (A_mean - B_mean) / max(
                        abs(B_mean),
                        1e-9,
                    )

                    if relative_difference >= MIN_RELATIVE_EDGE:
                        selected = "A"

                elif B_mean > A_mean:
                    relative_difference = (B_mean - A_mean) / max(
                        abs(A_mean),
                        1e-9,
                    )

                    if relative_difference >= MIN_RELATIVE_EDGE:
                        selected = "B"

            # ------------------------------------------------
            # Only A positive:
            # ------------------------------------------------

            elif A_mean > MIN_EXPECTANCY and B_mean <= MIN_EXPECTANCY:
                selected = "A"

            # ------------------------------------------------
            # Only B positive:
            # ------------------------------------------------

            elif B_mean > MIN_EXPECTANCY and A_mean <= MIN_EXPECTANCY:
                selected = "B"

        # ----------------------------------------------------
        # If only one candidate has enough observations,
        # we DO NOT automatically use it.
        #
        # This avoids allowing small-sample regimes to
        # dominate the router.
        # ----------------------------------------------------

        decisions.append(
            {
                "volatility": bucket,
                "selected": selected,
                "A_trades": A["trades"],
                "A_WR": A["WR"],
                "A_mean_R": A["mean_R"],
                "A_PF": A["PF"],
                "B_trades": B["trades"],
                "B_WR": B["WR"],
                "B_mean_R": B["mean_R"],
                "B_PF": B["PF"],
            }
        )

    return pd.DataFrame(decisions)


# ============================================================
# APPLY ROUTER
# ============================================================


def apply_router(
    candidate_trades,
    decisions,
):

    if candidate_trades.empty:
        return candidate_trades.copy()

    decision_map = dict(
        zip(
            decisions["volatility"],
            decisions["selected"],
        )
    )

    trades = candidate_trades.copy()

    trades["selected_model"] = trades["vol_bucket"].map(decision_map)

    trades["take_trade"] = trades["candidate"] == trades["selected_model"]

    return trades.loc[trades["take_trade"]].copy()


# ============================================================
# DAILY METRICS
# ============================================================


def daily_metrics(trades):

    if trades.empty:
        return {
            "trading_days": 0,
            "profitable_days": 0,
            "losing_days": 0,
            "mean_daily_R": np.nan,
            "median_daily_R": np.nan,
            "worst_day_R": np.nan,
            "best_day_R": np.nan,
            "trades_per_day": np.nan,
        }

    x = trades.copy()

    x["date"] = x["entry_timestamp"].dt.date

    daily = x.groupby("date")["net_R"].sum()

    return {
        "trading_days": len(daily),
        "profitable_days": int((daily > 0).sum()),
        "losing_days": int((daily < 0).sum()),
        "mean_daily_R": float(daily.mean()),
        "median_daily_R": float(daily.median()),
        "worst_day_R": float(daily.min()),
        "best_day_R": float(daily.max()),
        "trades_per_day": float(len(x) / len(daily)),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 RELATIVE REGIME ROUTER")

    print("=" * 110)

    print("\nFROZEN CANDIDATES:")

    print("A = Quality >= 0.65 | RR 1.25")

    print("B = Quality >= 0.75 | RR 1.30")

    print("\nROUTING:")

    print("A/B compared inside each volatility regime")

    print("TRAIN determines routing")

    print("OOS only executes frozen routing")

    print("\nNO XGBOOST.")

    print("NO strategy parameter optimization.")

    print("\nRouter requirements:")

    print(f"Minimum train trades: {MIN_TRAIN_TRADES}")

    print(f"Minimum relative edge: {MIN_RELATIVE_EDGE:.2f}")

    print(f"Minimum expectancy: {MIN_EXPECTANCY:.4f}R")

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    df = prepare_rth(df)

    df = add_directional_features(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    windows = generate_windows(df)

    print(f"Walk-forward windows: {len(windows)}")

    all_router_trades = []

    all_window_results = []

    all_routing_decisions = []

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for (
        window_number,
        (
            train_start,
            validation_start,
            validation_end,
        ),
    ) in enumerate(
        windows,
        start=1,
    ):
        print("\n" + "#" * 110)

        print(f"WINDOW {window_number}")

        print("#" * 110)

        # ----------------------------------------------------
        # TRAIN / OOS
        # ----------------------------------------------------

        train = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        # ----------------------------------------------------
        # HMM
        # ----------------------------------------------------

        train, validation = fit_hmm(
            train,
            validation,
        )

        # ----------------------------------------------------
        # TRAIN-ONLY STRATEGY THRESHOLDS
        # ----------------------------------------------------

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        # ----------------------------------------------------
        # TRAIN-BASED VOLATILITY PERCENTILES
        # ----------------------------------------------------

        train, validation = add_volatility_percentile(
            train,
            validation,
        )

        # ====================================================
        # GENERATE TRAIN TRADES
        # ====================================================

        train_trades_list = []

        for candidate in CANDIDATES:
            t = generate_candidate_trades(
                train,
                thresholds,
                scales,
                candidate,
            )

            if not t.empty:
                t = assign_trade_volatility(
                    t,
                    train,
                )

                train_trades_list.append(t)

        if train_trades_list:
            train_trades = pd.concat(
                train_trades_list,
                ignore_index=True,
            )

        else:
            train_trades = pd.DataFrame()

        # ====================================================
        # TRAIN ROUTER
        # ====================================================

        decisions = select_relative_models(train_trades)

        decisions["window"] = window_number

        all_routing_decisions.append(decisions)

        print("\nTRAIN ROUTING DECISIONS")

        print(
            decisions[
                [
                    "volatility",
                    "selected",
                    "A_trades",
                    "A_mean_R",
                    "A_PF",
                    "B_trades",
                    "B_mean_R",
                    "B_PF",
                ]
            ].to_string(index=False)
        )

        # ====================================================
        # GENERATE OOS CANDIDATE TRADES
        # ====================================================

        validation_trades_list = []

        for candidate in CANDIDATES:
            t = generate_candidate_trades(
                validation,
                thresholds,
                scales,
                candidate,
            )

            if not t.empty:
                t = assign_trade_volatility(
                    t,
                    validation,
                )

                t["window"] = window_number

                validation_trades_list.append(t)

        if validation_trades_list:
            validation_trades = pd.concat(
                validation_trades_list,
                ignore_index=True,
            )

        else:
            validation_trades = pd.DataFrame()

        # ====================================================
        # ROUTE OOS
        # ====================================================

        router_trades = apply_router(
            validation_trades,
            decisions,
        )

        if not router_trades.empty:
            all_router_trades.append(router_trades)

        # ====================================================
        # WINDOW RESULT
        # ====================================================

        metrics = calculate_metrics(router_trades)

        daily = daily_metrics(router_trades)

        print("\nOOS RELATIVE ROUTER RESULT")

        print(f"Trades: {metrics['trades']}")

        print(f"WR: {metrics['WR']:.4f}")

        print(f"Mean R: {metrics['mean_R']:.4f}")

        print(f"Total R: {metrics['total_R']:.2f}")

        print(f"PF: {metrics['PF']:.3f}")

        print(f"Max DD: {metrics['max_DD_R']:.2f}R")

        print(f"Trades/day: {daily['trades_per_day']}")

        all_window_results.append(
            {
                "window": window_number,
                **metrics,
                **daily,
            }
        )

    # ========================================================
    # COMBINE
    # ========================================================

    if all_router_trades:
        router_trades = pd.concat(
            all_router_trades,
            ignore_index=True,
        )

    else:
        router_trades = pd.DataFrame()

    window_results = pd.DataFrame(all_window_results)

    routing_results = pd.concat(
        all_routing_decisions,
        ignore_index=True,
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS RELATIVE ROUTER")

    print("=" * 110)

    final_metrics = calculate_metrics(router_trades)

    final_daily = daily_metrics(router_trades)

    print(f"Trades: {final_metrics['trades']}")

    print(f"WR: {final_metrics['WR']:.4f}")

    print(f"Mean R: {final_metrics['mean_R']:.4f}")

    print(f"Total R: {final_metrics['total_R']:.2f}")

    print(f"PF: {final_metrics['PF']:.3f}")

    print(f"Max DD: {final_metrics['max_DD_R']:.2f}R")

    print(f"Worst streak: {final_metrics['worst_streak']}")

    print(f"Trading days: {final_daily['trading_days']}")

    print(f"Profitable days: {final_daily['profitable_days']}")

    print(f"Losing days: {final_daily['losing_days']}")

    print(f"Mean daily R: {final_daily['mean_daily_R']:.4f}")

    print(f"Median daily R: {final_daily['median_daily_R']:.4f}")

    print(f"Worst day R: {final_daily['worst_day_R']:.4f}")

    print(f"Best day R: {final_daily['best_day_R']:.4f}")

    print(f"Trades/day: {final_daily['trades_per_day']:.3f}")

    # ========================================================
    # ROUTING FREQUENCY
    # ========================================================

    print("\n" + "=" * 110)

    print("ROUTING FREQUENCY")

    print("=" * 110)

    print(
        routing_results[
            [
                "window",
                "volatility",
                "selected",
            ]
        ].to_string(index=False)
    )

    print("\nSelections:")

    print(routing_results["selected"].value_counts().to_string())

    # ========================================================
    # OOS PERFORMANCE BY VOLATILITY
    # ========================================================

    print("\n" + "=" * 110)

    print("OOS ROUTER PERFORMANCE BY VOLATILITY")

    print("=" * 110)

    if not router_trades.empty:
        regime_rows = []

        for (
            bucket,
            group,
        ) in router_trades.groupby(
            "vol_bucket",
            observed=False,
        ):
            m = calculate_metrics(group)

            regime_rows.append(
                {
                    "volatility": bucket,
                    **m,
                }
            )

        print(pd.DataFrame(regime_rows).to_string(index=False))

    # ========================================================
    # OOS PERFORMANCE BY SELECTED MODEL
    # ========================================================

    print("\n" + "=" * 110)

    print("OOS PERFORMANCE BY SELECTED MODEL")

    print("=" * 110)

    if not router_trades.empty:
        model_rows = []

        for (
            model_name,
            group,
        ) in router_trades.groupby("selected_model"):
            m = calculate_metrics(group)

            model_rows.append(
                {
                    "model": model_name,
                    **m,
                }
            )

        print(pd.DataFrame(model_rows).to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    router_trades.to_csv(
        RESULTS_DIR / "s2_relative_regime_router_trades.csv",
        index=False,
    )

    window_results.to_csv(
        RESULTS_DIR / "s2_relative_regime_router_windows.csv",
        index=False,
    )

    routing_results.to_csv(
        RESULTS_DIR / "s2_relative_regime_router_decisions.csv",
        index=False,
    )

    print("\n" + "=" * 110)

    print("RELATIVE REGIME ROUTER COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_relative_regime_router_trades.csv")

    print("s2_relative_regime_router_windows.csv")

    print("s2_relative_regime_router_decisions.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
