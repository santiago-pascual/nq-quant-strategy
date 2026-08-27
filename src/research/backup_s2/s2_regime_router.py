from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 REGIME ROUTER
# ============================================================
#
# TWO FROZEN STRATEGIES
#
# Candidate A:
#   quality >= 0.65
#   RR = 1.25
#
# Candidate B:
#   quality >= 0.75
#   RR = 1.30
#
# SHARED:
#   HMM State 2
#   17.5% lower-tail conditions
#   20 point stop
#   15 bar horizon
#
# ROUTING:
#
# For every walk-forward window:
#
#   1. Use TRAIN only to measure A and B by volatility regime.
#   2. Select A, B, or OFF for each volatility bucket.
#   3. Freeze those decisions.
#   4. Apply them to the OOS validation period.
#
# IMPORTANT:
#
# The OOS data NEVER determines the routing decision.
#
# ============================================================


RANDOM_STATE = 42

TARGET_STATE = 2
TAIL_PERCENT = 17.5

STOP_POINTS = 20.0
HORIZON = 15


# ============================================================
# FROZEN STRATEGIES
# ============================================================

CANDIDATES = {
    "A": {
        "quality_threshold": 0.65,
        "rr": 1.25,
    },
    "B": {
        "quality_threshold": 0.75,
        "rr": 1.30,
    },
}


# ============================================================
# COST MODEL
# ============================================================

MNQ_POINT_VALUE = 2.00

TOPSTEP_MNQ_RT_FEE_USD = 1.22

SLIPPAGE_POINTS_PER_SIDE = 0.25

TOTAL_SLIPPAGE_POINTS = 2.0 * SLIPPAGE_POINTS_PER_SIDE

TOPSTEP_FEE_POINTS = TOPSTEP_MNQ_RT_FEE_USD / MNQ_POINT_VALUE

TOTAL_COST_POINTS = TOTAL_SLIPPAGE_POINTS + TOPSTEP_FEE_POINTS

TOTAL_COST_R = TOTAL_COST_POINTS / STOP_POINTS


# ============================================================
# VOLATILITY BUCKETS
# ============================================================

VOL_BUCKETS = [
    (0.00, 0.20, "0-20"),
    (0.20, 0.40, "20-40"),
    (0.40, 0.60, "40-60"),
    (0.60, 0.80, "60-80"),
    (0.80, 1.00, "80-100"),
]


BASE_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


# ============================================================
# PREPARE RTH
# ============================================================


def prepare_rth(df):

    df = df.copy()

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")

    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = timestamps

    df = df.loc[df["market_period"] == "RTH"].copy()

    df = df.sort_values("_timestamp_et")

    df = df.set_index("_timestamp_et")

    df.index.name = "timestamp_et"

    if "session_date" in df.columns:
        df["_session_id"] = df["session_date"].astype(str)

    else:
        df["_session_id"] = df.index.date

    return df


# ============================================================
# WALK FORWARD
# ============================================================


def generate_windows(df):

    start = df.index.min()
    end = df.index.max()

    validation_start = start + pd.DateOffset(years=2)

    windows = []

    while validation_start < end:
        validation_end = min(
            validation_start + pd.DateOffset(months=3),
            end,
        )

        train_start = validation_start - pd.DateOffset(years=2)

        windows.append(
            (
                train_start,
                validation_start,
                validation_end,
            )
        )

        validation_start += pd.DateOffset(months=3)

    return windows


# ============================================================
# HMM
# ============================================================


def fit_hmm(
    train,
    validation,
):

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
# THRESHOLDS
# ============================================================


def calculate_thresholds(train):

    state_train = train.loc[train["hmm_state"] == TARGET_STATE]

    thresholds = {}

    q = TAIL_PERCENT / 100.0

    for feature in BASE_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        thresholds[feature] = float(values.quantile(q))

    return thresholds


# ============================================================
# QUALITY SCALES
# ============================================================


def calculate_quality_scales(
    train,
):

    state_train = train.loc[train["hmm_state"] == TARGET_STATE]

    scales = {}

    for feature in BASE_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        threshold = values.quantile(TAIL_PERCENT / 100.0)

        extreme = values.quantile(0.05)

        scale = threshold - extreme

        if scale <= 0:
            scale = np.nan

        scales[feature] = float(scale)

    return scales


# ============================================================
# QUALITY
# ============================================================


def calculate_quality(
    row,
    thresholds,
    scales,
):

    scores = []

    for feature in BASE_FEATURES:
        value = row[feature]

        threshold = thresholds[feature]

        scale = scales[feature]

        if pd.isna(value) or pd.isna(scale) or scale <= 0:
            return np.nan

        score = (threshold - value) / scale

        score = np.clip(
            score,
            0.0,
            1.0,
        )

        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# BASE SETUP
# ============================================================


def base_setup_mask(
    df,
    thresholds,
):

    mask = df["hmm_state"] == TARGET_STATE

    for feature in BASE_FEATURES:
        mask &= df[feature] <= thresholds[feature]

    return mask


# ============================================================
# VOLATILITY PERCENTILE
# ============================================================


def add_volatility_percentile(
    train,
    validation,
):

    train = train.copy()
    validation = validation.copy()

    # IMPORTANT:
    #
    # Percentiles are learned from TRAIN.
    #
    # Validation is transformed using the
    # TRAIN volatility distribution.
    #

    train_values = (
        train["realized_vol_30"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .sort_values()
        .to_numpy()
    )

    def transform(series):

        values = series.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        result = np.full(
            len(values),
            np.nan,
            dtype=float,
        )

        valid = values.notna()

        if len(train_values) > 0:
            result[valid.to_numpy()] = np.searchsorted(
                train_values,
                values[valid].to_numpy(),
                side="right",
            ) / len(train_values)

        return result

    train["vol_percentile"] = transform(train["realized_vol_30"])

    validation["vol_percentile"] = transform(validation["realized_vol_30"])

    return (
        train,
        validation,
    )


# ============================================================
# VOLATILITY BUCKET
# ============================================================


def volatility_bucket(
    percentile,
):

    if pd.isna(percentile):
        return None

    for (
        low,
        high,
        label,
    ) in VOL_BUCKETS:
        if percentile >= low and percentile < high:
            return label

    if percentile >= 1.0:
        return "80-100"

    return None


# ============================================================
# RAW SHORT TRADE
# ============================================================


def resolve_short_trade(
    session,
    entry_position,
    rr,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = float(close[entry_position])

    target_points = STOP_POINTS * rr

    target_price = entry_price - target_points

    stop_price = entry_price + STOP_POINTS

    last_position = min(
        entry_position + HORIZON,
        len(session) - 1,
    )

    for i in range(
        entry_position + 1,
        last_position + 1,
    ):
        target_hit = low[i] <= target_price

        stop_hit = high[i] >= stop_price

        if target_hit and stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        if target_hit:
            return {
                "raw_points": target_points,
                "reason": "target",
                "exit_position": i,
            }

        if stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "stop",
                "exit_position": i,
            }

    exit_price = float(close[last_position])

    return {
        "raw_points": entry_price - exit_price,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# GENERATE CANDIDATE TRADES
# ============================================================


def generate_candidate_trades(
    df,
    thresholds,
    scales,
    candidate_name,
):

    params = CANDIDATES[candidate_name]

    quality_threshold = params["quality_threshold"]

    rr = params["rr"]

    records = []

    for (
        session_id,
        session,
    ) in df.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= HORIZON:
            continue

        valid = base_setup_mask(
            session,
            thresholds,
        )

        positions = np.flatnonzero(valid.to_numpy())

        i = 0

        while i < len(positions):
            position = positions[i]

            if position >= len(session) - HORIZON:
                break

            row = session.iloc[position]

            quality = calculate_quality(
                row,
                thresholds,
                scales,
            )

            if pd.isna(quality) or quality < quality_threshold:
                i += 1
                continue

            result = resolve_short_trade(
                session,
                position,
                rr,
            )

            raw_points = result["raw_points"]

            net_points = raw_points - TOTAL_COST_POINTS

            net_R = net_points / STOP_POINTS

            records.append(
                {
                    "entry_timestamp": session.index[position],
                    "exit_timestamp": session.index[result["exit_position"]],
                    "session_id": session_id,
                    "candidate": candidate_name,
                    "quality": quality,
                    "rr": rr,
                    "raw_points": raw_points,
                    "net_points": net_points,
                    "net_R": net_R,
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - position),
                }
            )

            exit_position = result["exit_position"]

            future_positions = positions[positions > exit_position]

            if not len(future_positions):
                break

            i = np.searchsorted(
                positions,
                future_positions[0],
            )

    return pd.DataFrame(records)


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades,
):

    if trades.empty:
        return {
            "trades": 0,
            "WR": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "PF": np.nan,
            "max_DD_R": np.nan,
            "worst_streak": 0,
        }

    pnl = trades["net_R"].astype(float)

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    PF = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    longest = 0
    current = 0

    for value in pnl:
        if value < 0:
            current += 1

            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    return {
        "trades": len(trades),
        "WR": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "total_R": float(pnl.sum()),
        "PF": float(PF),
        "max_DD_R": float(drawdown.min()),
        "worst_streak": int(longest),
    }


# ============================================================
# TRAIN REGIME PERFORMANCE
# ============================================================
#
# This determines which candidate gets used.
#
# IMPORTANT:
# ONLY TRAIN DATA is used.
#
# Selection rule:
#
#   Candidate must have:
#
#       mean_R > 0
#       PF > 1
#       minimum trade count
#
# If both qualify:
#
#       choose higher mean_R
#
# If neither qualifies:
#
#       OFF
#
# ============================================================

MIN_TRAIN_TRADES = 30


def select_regime_models(
    train_trades,
    train,
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

        valid_candidates = []

        for candidate, m in bucket_results.items():
            if m["trades"] >= MIN_TRAIN_TRADES and m["mean_R"] > 0 and m["PF"] > 1.0:
                valid_candidates.append(candidate)

        if not valid_candidates:
            selected = "OFF"

        else:
            selected = max(
                valid_candidates,
                key=lambda c: bucket_results[c]["mean_R"],
            )

        row = {
            "volatility": bucket,
            "selected": selected,
        }

        for candidate in CANDIDATES:
            m = bucket_results[candidate]

            row[f"{candidate}_trades"] = m["trades"]

            row[f"{candidate}_WR"] = m["WR"]

            row[f"{candidate}_mean_R"] = m["mean_R"]

            row[f"{candidate}_PF"] = m["PF"]

        decisions.append(row)

    return pd.DataFrame(decisions)


# ============================================================
# APPLY ROUTER
# ============================================================


def apply_router(
    validation_trades,
    decisions,
):

    decision_map = dict(
        zip(
            decisions["volatility"],
            decisions["selected"],
        )
    )

    trades = validation_trades.copy()

    trades["selected_model"] = trades["vol_bucket"].map(decision_map)

    trades["take_trade"] = trades["candidate"] == trades["selected_model"]

    return trades


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 REGIME ROUTER")

    print("=" * 110)

    print("\nFROZEN CANDIDATES:")

    print("A = Quality >= 0.65 | RR 1.25")

    print("B = Quality >= 0.75 | RR 1.30")

    print("\nROUTER:")

    print("TRAIN decides A / B / OFF")

    print("OOS only executes the frozen decision")

    print("\nExecution cost:")

    print(
        f"{TOTAL_COST_POINTS:.2f} points "
        f"= {TOTAL_COST_R:.4f}R "
        f"= ${TOTAL_COST_POINTS * MNQ_POINT_VALUE:.2f}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    df = prepare_rth(df)

    df = add_directional_features(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    windows = generate_windows(df)

    print(f"Walk-forward windows: {len(windows)}")

    all_router_trades = []
    all_decisions = []
    all_window_results = []

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
        # FEATURES / THRESHOLDS
        # ----------------------------------------------------

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        # ----------------------------------------------------
        # VOLATILITY PERCENTILES
        # ----------------------------------------------------

        (
            train,
            validation,
        ) = add_volatility_percentile(
            train,
            validation,
        )

        # ----------------------------------------------------
        # GENERATE TRAIN CANDIDATE TRADES
        # ----------------------------------------------------

        train_candidate_trades = []

        for candidate in CANDIDATES:
            t = generate_candidate_trades(
                train,
                thresholds,
                scales,
                candidate,
            )

            if not t.empty:
                train_candidate_trades.append(t)

        if train_candidate_trades:
            train_candidate_trades = pd.concat(
                train_candidate_trades,
                ignore_index=True,
            )

        else:
            train_candidate_trades = pd.DataFrame()

        if not train_candidate_trades.empty:
            train_candidate_trades["vol_bucket"] = train_candidate_trades[
                "entry_timestamp"
            ].map(validation.index.to_series())

            # The mapping above is not needed for
            # selection; assign volatility using
            # the TRAIN data directly.

            train_vol = train[["vol_percentile"]].copy()

            train_candidate_trades["vol_percentile"] = (
                train_vol["vol_percentile"]
                .reindex(
                    train_candidate_trades["entry_timestamp"],
                    method="ffill",
                )
                .to_numpy()
            )

            train_candidate_trades["vol_bucket"] = train_candidate_trades[
                "vol_percentile"
            ].apply(volatility_bucket)

        # ----------------------------------------------------
        # TRAIN ROUTING DECISION
        # ----------------------------------------------------

        decisions = select_regime_models(
            train_candidate_trades,
            train,
        )

        decisions["window"] = window_number

        all_decisions.append(decisions)

        print("\nTRAIN REGIME DECISIONS")

        print(decisions.to_string(index=False))

        # ----------------------------------------------------
        # GENERATE OOS TRADES
        # ----------------------------------------------------

        validation_candidate_trades = []

        for candidate in CANDIDATES:
            t = generate_candidate_trades(
                validation,
                thresholds,
                scales,
                candidate,
            )

            if not t.empty:
                validation_candidate_trades.append(t)

        if validation_candidate_trades:
            validation_candidate_trades = pd.concat(
                validation_candidate_trades,
                ignore_index=True,
            )

        else:
            validation_candidate_trades = pd.DataFrame()

        # ----------------------------------------------------
        # VALIDATION VOLATILITY
        # ----------------------------------------------------

        if not validation_candidate_trades.empty:
            validation_candidate_trades["vol_percentile"] = (
                validation[["vol_percentile"]]
                .reindex(
                    validation_candidate_trades["entry_timestamp"],
                    method="ffill",
                )
                .to_numpy()
            )

            validation_candidate_trades["vol_bucket"] = validation_candidate_trades[
                "vol_percentile"
            ].apply(volatility_bucket)

            # ------------------------------------------------
            # ROUTE
            # ------------------------------------------------

            routed = apply_router(
                validation_candidate_trades,
                decisions,
            )

            routed = routed.loc[routed["take_trade"]].copy()

        else:
            routed = pd.DataFrame()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if not routed.empty:
            routed["window"] = window_number

            all_router_trades.append(routed)

        # ----------------------------------------------------
        # WINDOW RESULT
        # ----------------------------------------------------

        m = calculate_metrics(routed)

        print("\nOOS ROUTER RESULT")

        print(f"Trades: {m['trades']}")

        print(f"WR: {m['WR']:.4f}")

        print(f"Mean R: {m['mean_R']:.4f}")

        print(f"Total R: {m['total_R']:.2f}")

        print(f"PF: {m['PF']:.3f}")

        print(f"Max DD: {m['max_DD_R']:.2f}R")

        all_window_results.append(
            {
                "window": window_number,
                **m,
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

    decisions_all = pd.concat(
        all_decisions,
        ignore_index=True,
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS REGIME ROUTER")

    print("=" * 110)

    final_metrics = calculate_metrics(router_trades)

    print(f"Trades: {final_metrics['trades']}")

    print(f"WR: {final_metrics['WR']:.4f}")

    print(f"Mean R: {final_metrics['mean_R']:.4f}")

    print(f"Total R: {final_metrics['total_R']:.2f}")

    print(f"PF: {final_metrics['PF']:.3f}")

    print(f"Max DD: {final_metrics['max_DD_R']:.2f}R")

    print(f"Worst streak: {final_metrics['worst_streak']}")

    # ========================================================
    # MODEL USAGE
    # ========================================================

    print("\n" + "=" * 110)

    print("MODEL USAGE")

    print("=" * 110)

    if not router_trades.empty:
        print(router_trades["selected_model"].value_counts().to_string())

    # ========================================================
    # REGIME RESULTS
    # ========================================================

    print("\n" + "=" * 110)

    print("OOS PERFORMANCE BY VOLATILITY REGIME")

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
    # DAILY PROFILE
    # ========================================================

    print("\n" + "=" * 110)

    print("DAILY PROFILE")

    print("=" * 110)

    if not router_trades.empty:
        temp = router_trades.copy()

        temp["date"] = temp["entry_timestamp"].dt.date

        daily = temp.groupby("date")["net_R"].sum()

        print(f"Trading days: {len(daily)}")

        print(f"Profitable days: {(daily > 0).sum()}")

        print(f"Losing days: {(daily < 0).sum()}")

        print(f"Mean daily R: {daily.mean():.4f}")

        print(f"Median daily R: {daily.median():.4f}")

        print(f"Worst day R: {daily.min():.4f}")

        print(f"Best day R: {daily.max():.4f}")

        print(f"Trades/day: {len(temp) / len(daily):.3f}")

    # ========================================================
    # SAVE
    # ========================================================

    router_trades.to_csv(
        "s2_regime_router_trades.csv",
        index=False,
    )

    decisions_all.to_csv(
        "s2_regime_router_decisions.csv",
        index=False,
    )

    window_results.to_csv(
        "s2_regime_router_windows.csv",
        index=False,
    )

    print("\nSaved:")

    print("s2_regime_router_trades.csv")

    print("s2_regime_router_decisions.csv")

    print("s2_regime_router_windows.csv")

    print("\n" + "=" * 110)

    print("REGIME ROUTER TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
