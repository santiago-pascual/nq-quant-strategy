from __future__ import annotations

from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08P
# FULL 22-WINDOW CONFIRMATION
# =============================================================================
#
# Purpose:
#   Re-run the THREE FROZEN candidates over ALL Research 07 temporal windows.
#
# IMPORTANT:
#   This is NOT a new optimization.
#   This is NOT an OOS test.
#   This is a historical/full-sample confirmation.
#
# Frozen candidates:
#
#   MRS2 = SHORT | HMM 2 | VOL 80-100 | Z 2.0 | TP 5 | SL 2 | H 5
#   MRL1 = LONG  | HMM 1 | VOL 20-40  | Z 2.5 | TP 5 | SL 2 | H 20
#   MRL2 = LONG  | HMM 2 | VOL 60-80  | Z 3.5 | TP 5 | SL 2 | H 2
#
# No:
#   - HMM retraining
#   - parameter optimization
#   - volatility optimization
#   - candidate selection
#   - failure filtering
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

OUTPUT_TRADES = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08p_full_confirmation_windows.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08p_full_confirmation_summary.csv"


# =============================================================================
# FROZEN CANDIDATES
# =============================================================================

CANDIDATES = [
    {
        "strategy_name": "MRS2",
        "candidate_id": "C01",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 5,
    },
    {
        "strategy_name": "MRL1",
        "candidate_id": "C02",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 20,
    },
    {
        "strategy_name": "MRL2",
        "candidate_id": "C06",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 2,
    },
]


# =============================================================================
# HELPERS
# =============================================================================


def section(title: str):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def normalize_timestamp(series):

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")


# =============================================================================
# LOAD RESEARCH 07 METADATA
# =============================================================================


def load_metadata():

    section("LOADING RESEARCH 07 METADATA")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    required = [
        "event_id",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing metadata columns: {missing}")

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(np.int16)

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df["zscore_30"] = pd.to_numeric(
        df["zscore_30"],
        errors="coerce",
    )

    df["timestamp"] = normalize_timestamp(df["timestamp"])

    df = df.sort_values("event_id").reset_index(drop=True)

    print(f"Events: {len(df):,}")

    print(f"Windows: {df['window'].nunique()}")

    print(f"First timestamp: {df['timestamp'].min()}")

    print(f"Last timestamp: {df['timestamp'].max()}")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm():

    section("LOADING RESEARCH 08B HMM")

    if not HMM_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{HMM_PATH}")

    df = pd.read_csv(HMM_PATH)

    required = [
        "event_id",
        "hmm_state",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing HMM columns: {missing}")

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(
        df["hmm_state"],
        errors="raise",
    ).astype(np.int8)

    print(f"HMM rows: {len(df):,}")

    return df[
        [
            "event_id",
            "hmm_state",
        ]
    ]


# =============================================================================
# BUILD CAUSAL VOLATILITY
# =============================================================================


def build_volatility(metadata):

    section("BUILDING CAUSAL VOLATILITY CONTEXT")

    from src.databento_loader import (
        load_databento_mnq,
    )

    from src.feature_engine import (
        add_return_features,
        add_volatility_features,
    )

    market = load_databento_mnq()

    print(f"Rows loaded: {len(market):,}")

    market = add_return_features(market)

    market = add_volatility_features(market)

    market["_timestamp"] = normalize_timestamp(market["timestamp ET"])

    market["realized_vol_30"] = pd.to_numeric(
        market["realized_vol_30"],
        errors="coerce",
    )

    market = market[market["_timestamp"].notna() & market["realized_vol_30"].notna()][
        [
            "_timestamp",
            "realized_vol_30",
        ]
    ].copy()

    market = (
        market.sort_values("_timestamp")
        .drop_duplicates(
            "_timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    event_times = metadata[
        [
            "event_id",
            "timestamp",
        ]
    ].copy()

    event_times = event_times.sort_values("timestamp").reset_index(drop=True)

    mapped = pd.merge_asof(
        event_times,
        market,
        left_on="timestamp",
        right_on="_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    mapped = mapped.sort_values("event_id").reset_index(drop=True)

    values = mapped["realized_vol_30"].to_numpy(dtype=np.float64)

    percentile = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    history = []

    valid = np.flatnonzero(np.isfinite(values))

    print(f"Valid volatility observations: {len(valid):,}")

    for counter, idx in enumerate(valid):
        value = float(values[idx])

        if history:
            pos = bisect_right(
                history,
                value,
            )

            percentile[idx] = pos / len(history) * 100.0

            history.insert(
                pos,
                value,
            )

        else:
            history.append(value)

        if (counter + 1) % 100_000 == 0 or counter + 1 == len(valid):
            print(f"  Percentile: {counter + 1:,}/{len(valid):,}")

    mapped["vol_percentile"] = percentile

    mapped["vol_bucket"] = np.select(
        [
            percentile < 20,
            percentile < 40,
            percentile < 60,
            percentile < 80,
            percentile >= 80,
        ],
        [
            "0-20",
            "20-40",
            "40-60",
            "60-80",
            "80-100",
        ],
        default="UNKNOWN",
    )

    return mapped[
        [
            "event_id",
            "realized_vol_30",
            "vol_percentile",
            "vol_bucket",
        ]
    ]


# =============================================================================
# BUILD COMPLETE EVENT TABLE
# =============================================================================


def build_events():

    section("BUILDING COMPLETE EVENT CONTEXT")

    metadata = load_metadata()

    hmm = load_hmm()

    volatility = build_volatility(metadata)

    events = metadata.merge(
        hmm,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    events = events.merge(
        volatility,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    missing_hmm = int(events["hmm_state"].isna().sum())

    missing_vol = int(events["vol_bucket"].isna().sum())

    missing_z = int(events["zscore_30"].isna().sum())

    print(f"Events: {len(events):,}")

    print(f"Missing HMM: {missing_hmm}")

    print(f"Missing volatility: {missing_vol}")

    print(f"Missing z-score: {missing_z}")

    if missing_hmm:
        raise RuntimeError("Missing HMM states.")

    if missing_z:
        raise RuntimeError("Missing z-score.")

    unknown = events["vol_bucket"] == "UNKNOWN"

    unknown_count = int(unknown.sum())

    if unknown_count:
        print(f"Removing {unknown_count} undefined initial volatility event(s).")

        events = events.loc[~unknown].copy()

    return events


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_path_cache():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{PATH_CACHE_PATH}")

    cache = np.load(
        PATH_CACHE_PATH,
        allow_pickle=False,
    )

    required = [
        "future_close",
    ]

    for key in required:
        if key not in cache.files:
            raise RuntimeError(f"Missing cache key: {key}")

    future_close = cache["future_close"]

    print(f"future_close: {future_close.shape}")

    if future_close.shape[1] < 120:
        raise RuntimeError("Insufficient path-cache horizon.")

    return future_close


# =============================================================================
# CONTEXT MASK
# =============================================================================


def candidate_mask(
    events,
    candidate,
):

    mask = events["hmm_state"] == candidate["hmm_state"]

    mask &= events["vol_bucket"] == candidate["vol_bucket"]

    z = candidate["zscore"]

    if candidate["side"] == "LONG":
        mask &= events["zscore_30"] <= -z

    else:
        mask &= events["zscore_30"] >= z

    return mask


# =============================================================================
# TRADE EVALUATION
# =============================================================================


def evaluate_trade(
    entry,
    future_path,
    candidate,
):

    side = candidate["side"]

    tp = candidate["tp"]
    sl = candidate["sl"]

    horizon = candidate["horizon"]

    path = future_path[:horizon]

    path = path[np.isfinite(path)]

    if len(path) == 0:
        return (
            "UNRESOLVED",
            0.0,
            np.nan,
        )

    if side == "LONG":
        movement = path - entry

    else:
        movement = entry - path

    tp_hits = np.flatnonzero(movement >= tp)

    sl_hits = np.flatnonzero(movement <= -sl)

    first_tp = int(tp_hits[0]) if len(tp_hits) else None

    first_sl = int(sl_hits[0]) if len(sl_hits) else None

    if first_tp is not None and first_sl is not None:
        if first_tp < first_sl:
            return (
                "WIN",
                candidate["rr"],
                first_tp + 1,
            )

        if first_sl < first_tp:
            return (
                "LOSS",
                -1.0,
                first_sl + 1,
            )

        # Same future_close observation reached
        # both thresholds.
        #
        # Intrabar order is unknowable.
        # Conservative treatment.
        return (
            "AMBIGUOUS",
            0.0,
            first_tp + 1,
        )

    if first_tp is not None:
        return (
            "WIN",
            candidate["rr"],
            first_tp + 1,
        )

    if first_sl is not None:
        return (
            "LOSS",
            -1.0,
            first_sl + 1,
        )

    return (
        "UNRESOLVED",
        0.0,
        horizon,
    )


# =============================================================================
# RUN ALL 22 WINDOWS
# =============================================================================


def run_confirmation(
    events,
    future_close,
):

    section("RUNNING FULL 22-WINDOW CONFIRMATION")

    windows = sorted(events["window"].unique().tolist())

    print(f"Temporal windows: {windows}")

    trades = []
    window_results = []

    for candidate in CANDIDATES:
        print("\n" + "-" * 100)

        print(
            f"{candidate['strategy_name']} | "
            f"{candidate['side']} | "
            f"HMM={candidate['hmm_state']} | "
            f"VOL={candidate['vol_bucket']} | "
            f"Z={candidate['zscore']} | "
            f"TP={candidate['tp']} | "
            f"SL={candidate['sl']} | "
            f"RR={candidate['rr']} | "
            f"H={candidate['horizon']}"
        )

        mask = candidate_mask(
            events,
            candidate,
        )

        candidate_events = events.loc[mask].copy()

        print(f"All-sample candidate observations: {len(candidate_events):,}")

        for window in windows:
            subset = candidate_events[candidate_events["window"] == window]

            if subset.empty:
                window_results.append(
                    {
                        "strategy_name": candidate["strategy_name"],
                        "candidate_id": candidate["candidate_id"],
                        "window": window,
                        "observations": 0,
                        "wins": 0,
                        "losses": 0,
                        "ambiguous": 0,
                        "unresolved": 0,
                        "resolved": 0,
                        "wr": np.nan,
                        "resolution": np.nan,
                        "net_r": 0.0,
                        "expectancy_all": np.nan,
                        "profit_factor": np.nan,
                    }
                )

                continue

            wins = 0
            losses = 0
            ambiguous = 0
            unresolved = 0
            net_r = 0.0

            for row in subset.itertuples(index=False):
                event_id = int(row.event_id)

                result, r, bars = evaluate_trade(
                    float(row.close),
                    future_close[event_id],
                    candidate,
                )

                if result == "WIN":
                    wins += 1

                elif result == "LOSS":
                    losses += 1

                elif result == "AMBIGUOUS":
                    ambiguous += 1

                elif result == "UNRESOLVED":
                    unresolved += 1

                net_r += r

                trades.append(
                    {
                        "strategy_name": candidate["strategy_name"],
                        "candidate_id": candidate["candidate_id"],
                        "side": candidate["side"],
                        "hmm_state": candidate["hmm_state"],
                        "vol_bucket": candidate["vol_bucket"],
                        "zscore": candidate["zscore"],
                        "tp": candidate["tp"],
                        "sl": candidate["sl"],
                        "rr": candidate["rr"],
                        "horizon": candidate["horizon"],
                        "event_id": event_id,
                        "window": int(row.window),
                        "timestamp": row.timestamp,
                        "entry": float(row.close),
                        "result": result,
                        "r": r,
                        "bars_to_result": bars,
                    }
                )

            observations = len(subset)

            resolved = wins + losses

            wr = wins / resolved if resolved else np.nan

            gross_profit = sum(
                max(
                    0.0,
                    x,
                )
                for x in [candidate["rr"]] * wins
            )

            gross_loss = float(losses)

            pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

            expectancy = net_r / observations if observations else np.nan

            window_results.append(
                {
                    "strategy_name": candidate["strategy_name"],
                    "candidate_id": candidate["candidate_id"],
                    "window": window,
                    "observations": observations,
                    "wins": wins,
                    "losses": losses,
                    "ambiguous": ambiguous,
                    "unresolved": unresolved,
                    "resolved": resolved,
                    "wr": wr,
                    "resolution": (resolved / observations if observations else np.nan),
                    "net_r": net_r,
                    "expectancy_all": expectancy,
                    "profit_factor": pf,
                }
            )

    return (
        pd.DataFrame(trades),
        pd.DataFrame(window_results),
    )


# =============================================================================
# SUMMARY
# =============================================================================


def build_summary(
    trades,
    windows,
):

    section("BUILDING FULL-SAMPLE SUMMARY")

    rows = []

    for candidate in CANDIDATES:
        name = candidate["strategy_name"]

        group = trades[trades["strategy_name"] == name]

        if group.empty:
            continue

        wins = int((group["result"] == "WIN").sum())

        losses = int((group["result"] == "LOSS").sum())

        ambiguous = int((group["result"] == "AMBIGUOUS").sum())

        unresolved = int((group["result"] == "UNRESOLVED").sum())

        observations = len(group)

        resolved = wins + losses

        wr = wins / resolved if resolved else np.nan

        net_r = float(group["r"].sum())

        expectancy = net_r / observations if observations else np.nan

        gross_profit = float(
            group.loc[
                group["r"] > 0,
                "r",
            ].sum()
        )

        gross_loss = float(
            -group.loc[
                group["r"] < 0,
                "r",
            ].sum()
        )

        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        wg = windows[windows["strategy_name"] == name]

        valid_windows = wg[wg["observations"] > 0]

        positive_windows = int((valid_windows["expectancy_all"] > 0).sum())

        negative_windows = int((valid_windows["expectancy_all"] < 0).sum())

        positive_wr_windows = int((valid_windows["wr"] >= 0.50).sum())

        rows.append(
            {
                "strategy_name": name,
                "candidate_id": candidate["candidate_id"],
                "side": candidate["side"],
                "hmm_state": candidate["hmm_state"],
                "vol_bucket": candidate["vol_bucket"],
                "zscore": candidate["zscore"],
                "tp": candidate["tp"],
                "sl": candidate["sl"],
                "rr": candidate["rr"],
                "horizon": candidate["horizon"],
                "observations": observations,
                "wins": wins,
                "losses": losses,
                "ambiguous": ambiguous,
                "unresolved": unresolved,
                "resolved": resolved,
                "wr": wr,
                "resolution": (resolved / observations if observations else np.nan),
                "net_r": net_r,
                "expectancy_all": expectancy,
                "profit_factor": pf,
                "positive_windows": positive_windows,
                "negative_windows": negative_windows,
                "valid_windows": len(valid_windows),
                "positive_wr_windows": positive_wr_windows,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08P")

    print("FULL 22-WINDOW CONFIRMATION")

    print("-" * 100)

    print("This is NOT an optimization.")

    print("This is NOT an OOS test.")

    print("All three candidates are frozen.")

    print("\nFROZEN CANDIDATES:")

    print("MRS2 = SHORT | HMM 2 | VOL 80-100 | Z 2.0 | TP 5 | SL 2 | H 5")

    print("MRL1 = LONG  | HMM 1 | VOL 20-40 | Z 2.5 | TP 5 | SL 2 | H 20")

    print("MRL2 = LONG  | HMM 2 | VOL 60-80 | Z 3.5 | TP 5 | SL 2 | H 2")

    events = build_events()

    future_close = load_path_cache()

    trades, windows = run_confirmation(
        events,
        future_close,
    )

    if trades.empty:
        raise RuntimeError("No confirmation trades generated.")

    summary = build_summary(
        trades,
        windows,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    section("FULL 22-WINDOW RESULTS")

    print(summary.to_string(index=False))

    section("WINDOW-BY-WINDOW RESULTS")

    display_columns = [
        "strategy_name",
        "window",
        "observations",
        "wins",
        "losses",
        "wr",
        "net_r",
        "expectancy_all",
        "profit_factor",
    ]

    print(windows[display_columns].to_string(index=False))

    section("TEMPORAL CONSISTENCY")

    for row in summary.itertuples(index=False):
        print(
            f"{row.strategy_name}: "
            f"WR={row.wr:.2%} | "
            f"Expectancy="
            f"{row.expectancy_all:.4f}R | "
            f"PF={row.profit_factor:.3f} | "
            f"Positive windows="
            f"{row.positive_windows}/"
            f"{row.valid_windows} | "
            f"Negative windows="
            f"{row.negative_windows}"
        )

    section("RESEARCH 08P COMPLETE")

    print("Frozen candidates were evaluated over all available temporal windows.")

    print("\nFILES SAVED:")

    print(OUTPUT_TRADES)

    print(OUTPUT_WINDOWS)

    print(OUTPUT_SUMMARY)

    print("\nNEXT STEP:")

    print(
        "If the full-history behavior is consistent "
        "with the independent OOS result, proceed "
        "to the frozen-candidate failure analysis."
    )


if __name__ == "__main__":
    main()
