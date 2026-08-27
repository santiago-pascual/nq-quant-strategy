from __future__ import annotations

import numpy as np
import pandas as pd

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
    select_regime_models,
    calculate_metrics,
    volatility_bucket,
)


# ============================================================
# S2 ARCHITECTURE COMPARISON
# ============================================================
#
# COMPARES:
#
#   1. Candidate A always
#   2. Candidate B always
#   3. A/B regime router
#   4. A/B/OFF regime router
#
# IMPORTANT:
#
#   - Parameters are frozen.
#   - No OOS optimization.
#   - Router decisions use TRAIN only.
#   - OOS is untouched when selecting the model.
#   - Same execution costs.
#   - Same non-overlapping execution.
#   - Same four walk-forward windows.
#
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

    vol = market_data[["vol_percentile"]].copy()

    trades["vol_percentile"] = (
        vol["vol_percentile"]
        .reindex(
            trades["entry_timestamp"],
            method="ffill",
        )
        .to_numpy()
    )

    trades["vol_bucket"] = trades["vol_percentile"].apply(volatility_bucket)

    return trades


# ============================================================
# ROUTER
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
            "days": 0,
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
        "days": len(daily),
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
    print("S2 ARCHITECTURE COMPARISON")
    print("=" * 110)

    print("\nFROZEN STRATEGIES:")

    print("A = Quality >= 0.65 | RR 1.25")

    print("B = Quality >= 0.75 | RR 1.30")

    print("\nARCHITECTURES:")

    print("1. A ALWAYS")

    print("2. B ALWAYS")

    print("3. A/B ROUTER")

    print("4. A/B/OFF ROUTER")

    print("\nNO PARAMETER OPTIMIZATION.")

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

    all_trades = []

    window_summary = []

    routing_history = []

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
        # TRAIN PARAMETERS
        # ----------------------------------------------------

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        # ----------------------------------------------------
        # VOLATILITY TRANSFORMATION
        #
        # TRAIN DISTRIBUTION ONLY.
        # ----------------------------------------------------

        train, validation = add_volatility_percentile(
            train,
            validation,
        )

        # ====================================================
        # TRAIN CANDIDATE TRADES
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
        # TRAIN ROUTING DECISION
        # ====================================================

        decisions = select_regime_models(
            train_trades,
            train,
        )

        decisions["window"] = window_number

        routing_history.append(decisions)

        print("\nTRAIN ROUTING DECISION")

        print(
            decisions[
                [
                    "volatility",
                    "selected",
                ]
            ].to_string(index=False)
        )

        # ====================================================
        # OOS CANDIDATE TRADES
        # ====================================================

        oos_trades_list = []

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

                oos_trades_list.append(t)

        if oos_trades_list:
            oos_trades = pd.concat(
                oos_trades_list,
                ignore_index=True,
            )

        else:
            oos_trades = pd.DataFrame()

        # ====================================================
        # ARCHITECTURE 1
        # A ALWAYS
        # ====================================================

        if not oos_trades.empty:
            A = oos_trades.loc[oos_trades["candidate"] == "A"].copy()

            A["architecture"] = "A_ALWAYS"

            all_trades.append(A)

        else:
            A = pd.DataFrame()

        # ====================================================
        # ARCHITECTURE 2
        # B ALWAYS
        # ====================================================

        if not oos_trades.empty:
            B = oos_trades.loc[oos_trades["candidate"] == "B"].copy()

            B["architecture"] = "B_ALWAYS"

            all_trades.append(B)

        else:
            B = pd.DataFrame()

        # ====================================================
        # ARCHITECTURE 3/4
        #
        # The router currently uses:
        #
        #   A
        #   B
        #   OFF
        #
        # We preserve the same TRAIN decisions.
        #
        # ====================================================

        router = apply_router(
            oos_trades,
            decisions,
        )

        if not router.empty:
            router = router.copy()

            router["architecture"] = "AB_ROUTER"

            all_trades.append(router)

            router_off = router.copy()

            router_off["architecture"] = "AB_OFF_ROUTER"

            all_trades.append(router_off)

        # ====================================================
        # WINDOW RESULTS
        # ====================================================

        print("\nOOS WINDOW COMPARISON")

        for (
            name,
            trades,
        ) in [
            ("A_ALWAYS", A),
            ("B_ALWAYS", B),
            ("AB_ROUTER", router),
        ]:
            m = calculate_metrics(trades)

            print(f"\n{name}")

            print(f"Trades: {m['trades']}")

            print(f"WR: {m['WR']:.4f}")

            print(f"Mean R: {m['mean_R']:.4f}")

            print(f"Total R: {m['total_R']:.2f}")

            print(f"PF: {m['PF']:.3f}")

            print(f"Max DD: {m['max_DD_R']:.2f}R")

            window_summary.append(
                {
                    "window": window_number,
                    "architecture": name,
                    **m,
                }
            )

    # ========================================================
    # COMBINE
    # ========================================================

    all_trades_df = pd.concat(
        all_trades,
        ignore_index=True,
    )

    window_summary_df = pd.DataFrame(window_summary)

    routing_df = pd.concat(
        routing_history,
        ignore_index=True,
    )

    # ========================================================
    # IMPORTANT:
    #
    # AB_ROUTER and AB_OFF_ROUTER have
    # identical trades in this implementation.
    #
    # They represent the same actual architecture:
    #
    #   A/B/OFF
    #
    # because OFF trades simply do not exist.
    #
    # ========================================================

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS ARCHITECTURE COMPARISON")

    print("=" * 110)

    final_rows = []

    for architecture in [
        "A_ALWAYS",
        "B_ALWAYS",
        "AB_ROUTER",
    ]:
        t = all_trades_df.loc[all_trades_df["architecture"] == architecture].copy()

        m = calculate_metrics(t)

        d = daily_metrics(t)

        positive_windows = int(
            (
                window_summary_df.loc[
                    window_summary_df["architecture"]
                    == ("AB_ROUTER" if architecture == "AB_ROUTER" else architecture),
                    "total_R",
                ]
                > 0
            ).sum()
        )

        final_rows.append(
            {
                "architecture": architecture,
                "trades": m["trades"],
                "WR": m["WR"],
                "mean_R": m["mean_R"],
                "total_R": m["total_R"],
                "PF": m["PF"],
                "max_DD_R": m["max_DD_R"],
                "worst_streak": m["worst_streak"],
                "trading_days": d["days"],
                "profitable_days": d["profitable_days"],
                "losing_days": d["losing_days"],
                "mean_daily_R": d["mean_daily_R"],
                "median_daily_R": d["median_daily_R"],
                "worst_day_R": d["worst_day_R"],
                "best_day_R": d["best_day_R"],
                "trades_per_day": d["trades_per_day"],
                "positive_windows": positive_windows,
                "total_windows": len(windows),
            }
        )

    final_df = pd.DataFrame(final_rows)

    print(final_df.to_string(index=False))

    # ========================================================
    # WINDOW-BY-WINDOW
    # ========================================================

    print("\n" + "=" * 110)

    print("WINDOW-BY-WINDOW RESULTS")

    print("=" * 110)

    print(window_summary_df.to_string(index=False))

    # ========================================================
    # ROUTING HISTORY
    # ========================================================

    print("\n" + "=" * 110)

    print("ROUTING HISTORY")

    print("=" * 110)

    print(
        routing_df[
            [
                "window",
                "volatility",
                "selected",
                "A_trades",
                "A_WR",
                "A_mean_R",
                "A_PF",
                "B_trades",
                "B_WR",
                "B_mean_R",
                "B_PF",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # VOLATILITY PERFORMANCE
    # ========================================================

    print("\n" + "=" * 110)

    print("ROUTER OOS PERFORMANCE BY VOLATILITY")

    print("=" * 110)

    router_trades = all_trades_df.loc[
        all_trades_df["architecture"] == "AB_ROUTER"
    ].copy()

    if not router_trades.empty:
        rows = []

        for (
            bucket,
            group,
        ) in router_trades.groupby(
            "vol_bucket",
            observed=False,
        ):
            m = calculate_metrics(group)

            rows.append(
                {
                    "volatility": bucket,
                    **m,
                }
            )

        print(pd.DataFrame(rows).to_string(index=False))

    # ========================================================
    # ROUTER MODEL USAGE
    # ========================================================

    print("\n" + "=" * 110)

    print("ROUTER MODEL USAGE")

    print("=" * 110)

    if not router_trades.empty:
        print(router_trades["selected_model"].value_counts().to_string())

    # ========================================================
    # DIRECT IMPROVEMENT VS A/B
    # ========================================================

    print("\n" + "=" * 110)

    print("ROUTER IMPROVEMENT")

    print("=" * 110)

    results = final_df.set_index("architecture")

    router_total = results.loc[
        "AB_ROUTER",
        "total_R",
    ]

    A_total = results.loc[
        "A_ALWAYS",
        "total_R",
    ]

    B_total = results.loc[
        "B_ALWAYS",
        "total_R",
    ]

    print(f"Router vs A: {router_total - A_total:+.2f}R")

    print(f"Router vs B: {router_total - B_total:+.2f}R")

    print(f"Router mean R: {results.loc['AB_ROUTER', 'mean_R']:.4f}")

    print(f"A mean R: {results.loc['A_ALWAYS', 'mean_R']:.4f}")

    print(f"B mean R: {results.loc['B_ALWAYS', 'mean_R']:.4f}")

    # ========================================================
    # SAVE
    # ========================================================

    final_df.to_csv(
        "s2_architecture_comparison_summary.csv",
        index=False,
    )

    window_summary_df.to_csv(
        "s2_architecture_comparison_windows.csv",
        index=False,
    )

    routing_df.to_csv(
        "s2_architecture_comparison_routing.csv",
        index=False,
    )

    all_trades_df.to_csv(
        "s2_architecture_comparison_trades.csv",
        index=False,
    )

    print("\n" + "=" * 110)

    print("ARCHITECTURE COMPARISON COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_architecture_comparison_summary.csv")

    print("s2_architecture_comparison_windows.csv")

    print("s2_architecture_comparison_routing.csv")

    print("s2_architecture_comparison_trades.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
