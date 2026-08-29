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
    VOL_BUCKETS,
    generate_windows,
    prepare_rth,
    calculate_thresholds,
    calculate_quality_scales,
    add_volatility_percentile,
    generate_candidate_trades,
    calculate_metrics,
    volatility_bucket,
    select_regime_models,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# S2 SELECTIVE EXECUTION
# ============================================================
#
# PURPOSE
# -------
# Test whether S2 is viable simply by NOT TRADING in regimes
# where the training data does not support the strategy.
#
#
# FROZEN STRATEGIES
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
#   17.5% lower-tail
#   20 point stop
#   15 bar horizon
#
#
# EXECUTION COST
# --------------
#
# MNQ:
#   $2 / point
#
# Topstep:
#   $1.22 round trip
#
# Slippage:
#   0.25 points / side
#
# Total:
#   1.11 points / trade
#   0.0555R / trade
#   $2.22 / trade
#
#
# IMPORTANT
# ---------
#
# The allowed regimes are NOT chosen from OOS performance.
#
# For each walk-forward window:
#
#   TRAIN -> determine which volatility regimes are usable
#   OOS   -> trade only those regimes
#
# This simulates what the strategy could have known
# before entering the OOS period.
#
#
# THREE SYSTEMS ARE TESTED
# ------------------------
#
# 1. A_SELECTIVE
#       Candidate A + only TRAIN-approved regimes
#
# 2. B_SELECTIVE
#       Candidate B + only TRAIN-approved regimes
#
# 3. AB_SELECTIVE
#       Whichever candidate the TRAIN router selects
#       in each regime.
#
#
# The purpose is to determine whether simply staying
# OUT of bad environments improves the strategy enough
# to be useful as a funded-account system.
#
# ============================================================


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
# ASSIGN VOLATILITY
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
# MONTHLY METRICS
# ============================================================


def monthly_metrics(trades):

    if trades.empty:
        return {
            "months": 0,
            "profitable_months": 0,
            "losing_months": 0,
            "mean_monthly_R": np.nan,
            "median_monthly_R": np.nan,
            "worst_month_R": np.nan,
            "best_month_R": np.nan,
        }

    x = trades.copy()

    x["month"] = x["entry_timestamp"].dt.to_period("M")

    monthly = x.groupby("month")["net_R"].sum()

    return {
        "months": len(monthly),
        "profitable_months": int((monthly > 0).sum()),
        "losing_months": int((monthly < 0).sum()),
        "mean_monthly_R": float(monthly.mean()),
        "median_monthly_R": float(monthly.median()),
        "worst_month_R": float(monthly.min()),
        "best_month_R": float(monthly.max()),
    }


# ============================================================
# APPLY FIXED REGIME MASK
# ============================================================


def apply_allowed_regimes(
    trades,
    allowed_regimes,
):

    if trades.empty:
        return trades.copy()

    return trades.loc[trades["vol_bucket"].isin(allowed_regimes)].copy()


# ============================================================
# APPLY TRAIN ROUTER
# ============================================================


def apply_train_router(
    trades,
    decisions,
):

    if trades.empty:
        return trades.copy()

    decision_map = dict(
        zip(
            decisions["volatility"],
            decisions["selected"],
        )
    )

    x = trades.copy()

    x["selected_model"] = x["vol_bucket"].map(decision_map)

    return x.loc[x["candidate"] == x["selected_model"]].copy()


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 SELECTIVE EXECUTION")

    print("=" * 110)

    print("\nFROZEN STRATEGIES:")

    print("A = Quality >= 0.65 | RR 1.25")

    print("B = Quality >= 0.75 | RR 1.30")

    print("\nTEST:")

    print("Trade only in TRAIN-approved volatility regimes.")

    print("Everything else = OFF.")

    print("\nNO PARAMETER OPTIMIZATION.")

    print("NO OOS REGIME SELECTION.")

    print("NO XGBOOST.")

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    df = prepare_rth(df)

    df = add_directional_features(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    windows = generate_windows(df)

    print(f"Walk-forward windows: {len(windows)}")

    # ========================================================
    # STORAGE
    # ========================================================

    all_A = []
    all_B = []
    all_AB = []

    window_rows = []
    routing_rows = []

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
        # FROZEN TRAIN-ONLY PARAMETERS
        # ----------------------------------------------------

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        # ----------------------------------------------------
        # VOLATILITY PERCENTILES
        # ----------------------------------------------------

        train, validation = add_volatility_percentile(
            train,
            validation,
        )

        # ====================================================
        # TRAIN TRADES
        # ====================================================

        train_list = []

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

                train_list.append(t)

        if train_list:
            train_trades = pd.concat(
                train_list,
                ignore_index=True,
            )

        else:
            train_trades = pd.DataFrame()

        # ====================================================
        # TRAIN ROUTING DECISION
        # ====================================================

        decisions = select_regime_models(
            train_trades,
            train,
        )

        decisions["window"] = window_number

        routing_rows.append(decisions)

        print("\nTRAIN REGIME MAP")

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

        # ----------------------------------------------------
        # APPROVED REGIMES
        # ----------------------------------------------------

        allowed_A = set(
            decisions.loc[
                decisions["selected"] == "A",
                "volatility",
            ]
        )

        allowed_B = set(
            decisions.loc[
                decisions["selected"] == "B",
                "volatility",
            ]
        )

        # For the "selective" version we also include
        # regimes where the router selected either A or B.
        #
        # This measures:
        #
        # "Does simply staying out of regimes rejected
        #  by TRAIN improve S2?"
        #
        allowed_any = allowed_A | allowed_B

        print("\nTRAIN-APPROVED REGIMES:")

        print(f"A selective: {sorted(allowed_A)}")

        print(f"B selective: {sorted(allowed_B)}")

        print(f"Any strategy: {sorted(allowed_any)}")

        # ====================================================
        # OOS TRADES
        # ====================================================

        oos_list = []

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

                oos_list.append(t)

        if oos_list:
            oos_trades = pd.concat(
                oos_list,
                ignore_index=True,
            )

        else:
            oos_trades = pd.DataFrame()

        # ====================================================
        # A SELECTIVE
        # ====================================================

        A_oos = oos_trades.loc[oos_trades["candidate"] == "A"].copy()

        A_selective = apply_allowed_regimes(
            A_oos,
            allowed_any,
        )

        # ====================================================
        # B SELECTIVE
        # ====================================================

        B_oos = oos_trades.loc[oos_trades["candidate"] == "B"].copy()

        B_selective = apply_allowed_regimes(
            B_oos,
            allowed_any,
        )

        # ====================================================
        # AB SELECTIVE
        #
        # Actual regime-specific model selection.
        # ====================================================

        AB_selective = apply_train_router(
            oos_trades,
            decisions,
        )

        # ====================================================
        # SAVE TRADES
        # ====================================================

        if not A_selective.empty:
            all_A.append(A_selective)

        if not B_selective.empty:
            all_B.append(B_selective)

        if not AB_selective.empty:
            all_AB.append(AB_selective)

        # ====================================================
        # WINDOW RESULTS
        # ====================================================

        for (
            name,
            trades,
        ) in [
            (
                "A_SELECTIVE",
                A_selective,
            ),
            (
                "B_SELECTIVE",
                B_selective,
            ),
            (
                "AB_SELECTIVE",
                AB_selective,
            ),
        ]:
            m = calculate_metrics(trades)

            d = daily_metrics(trades)

            mo = monthly_metrics(trades)

            print(f"\n{name}")

            print(f"Trades: {m['trades']}")

            print(f"WR: {m['WR']:.4f}")

            print(f"Mean R: {m['mean_R']:.4f}")

            print(f"Total R: {m['total_R']:.2f}")

            print(f"PF: {m['PF']:.3f}")

            print(f"Max DD: {m['max_DD_R']:.2f}R")

            window_rows.append(
                {
                    "window": window_number,
                    "architecture": name,
                    **m,
                    **d,
                    **mo,
                }
            )

    # ========================================================
    # COMBINE
    # ========================================================

    A_all = (
        pd.concat(
            all_A,
            ignore_index=True,
        )
        if all_A
        else pd.DataFrame()
    )

    B_all = (
        pd.concat(
            all_B,
            ignore_index=True,
        )
        if all_B
        else pd.DataFrame()
    )

    AB_all = (
        pd.concat(
            all_AB,
            ignore_index=True,
        )
        if all_AB
        else pd.DataFrame()
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS SELECTIVE EXECUTION")

    print("=" * 110)

    final_rows = []

    for (
        name,
        trades,
    ) in [
        (
            "A_SELECTIVE",
            A_all,
        ),
        (
            "B_SELECTIVE",
            B_all,
        ),
        (
            "AB_SELECTIVE",
            AB_all,
        ),
    ]:
        m = calculate_metrics(trades)

        d = daily_metrics(trades)

        mo = monthly_metrics(trades)

        final_rows.append(
            {
                "architecture": name,
                **m,
                **d,
                **mo,
            }
        )

    final_df = pd.DataFrame(final_rows)

    print(final_df.to_string(index=False))

    # ========================================================
    # WINDOW RESULTS
    # ========================================================

    print("\n" + "=" * 110)

    print("WINDOW RESULTS")

    print("=" * 110)

    window_df = pd.DataFrame(window_rows)

    print(window_df.to_string(index=False))

    # ========================================================
    # MONTHLY PERFORMANCE
    # ========================================================

    print("\n" + "=" * 110)

    print("MONTHLY PERFORMANCE")

    print("=" * 110)

    for (
        name,
        trades,
    ) in [
        (
            "A_SELECTIVE",
            A_all,
        ),
        (
            "B_SELECTIVE",
            B_all,
        ),
        (
            "AB_SELECTIVE",
            AB_all,
        ),
    ]:
        print(f"\n{name}")

        mo = monthly_metrics(trades)

        print(f"Months: {mo['months']}")

        print(f"Profitable months: {mo['profitable_months']}")

        print(f"Losing months: {mo['losing_months']}")

        print(f"Mean monthly R: {mo['mean_monthly_R']:.4f}")

        print(f"Median monthly R: {mo['median_monthly_R']:.4f}")

        print(f"Worst month R: {mo['worst_month_R']:.4f}")

        print(f"Best month R: {mo['best_month_R']:.4f}")

    # ========================================================
    # CAPITAL SCENARIOS
    # ========================================================
    #
    # Purely illustrative R multiples.
    #
    # We do NOT assume a specific funded-account risk.
    #
    # ========================================================

    print("\n" + "=" * 110)

    print("ILLUSTRATIVE R-SCALE")

    print("=" * 110)

    for (
        name,
        trades,
    ) in [
        (
            "A_SELECTIVE",
            A_all,
        ),
        (
            "B_SELECTIVE",
            B_all,
        ),
        (
            "AB_SELECTIVE",
            AB_all,
        ),
    ]:
        if trades.empty:
            continue

        mo = monthly_metrics(trades)

        print(f"\n{name}")

        print(f"Average monthly R: {mo['mean_monthly_R']:.3f}")

        print(f"Median monthly R: {mo['median_monthly_R']:.3f}")

        print(f"Worst month R: {mo['worst_month_R']:.3f}")

        print(f"Best month R: {mo['best_month_R']:.3f}")

    # ========================================================
    # SAVE
    # ========================================================

    final_df.to_csv(
        RESULTS_DIR / "s2_selective_execution_summary.csv",
        index=False,
    )

    window_df.to_csv(
        RESULTS_DIR / "s2_selective_execution_windows.csv",
        index=False,
    )

    A_all.to_csv(
        RESULTS_DIR / "s2_selective_execution_A_trades.csv",
        index=False,
    )

    B_all.to_csv(
        RESULTS_DIR / "s2_selective_execution_B_trades.csv",
        index=False,
    )

    AB_all.to_csv(
        RESULTS_DIR / "s2_selective_execution_AB_trades.csv",
        index=False,
    )

    routing_df = pd.concat(
        routing_rows,
        ignore_index=True,
    )

    routing_df.to_csv(
        RESULTS_DIR / "s2_selective_execution_routing.csv",
        index=False,
    )

    print("\n" + "=" * 110)

    print("SELECTIVE EXECUTION COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_selective_execution_summary.csv")

    print("s2_selective_execution_windows.csv")

    print("s2_selective_execution_A_trades.csv")

    print("s2_selective_execution_B_trades.csv")

    print("s2_selective_execution_AB_trades.csv")

    print("s2_selective_execution_routing.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
