"""
MEAN REVERSION — RESEARCH 08AB
==============================

LOCAL CANDIDATE CONFIRMATION

Reconstructs the 08Z candidate populations and evaluates the new
TP / SL / H configurations using canonical Research 07 events and
raw OHLC data.

IMPORTANT
---------
This script does NOT use the old 08P trade file.

It reconstructs:

    RTH data
    -> causal volatility percentile
    -> volatility bucket
    -> HMM state
    -> z-score signal
    -> future OHLC
    -> intrabar TP/SL resolution

Same-bar rule:
    STOP FIRST
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

EVENT_FILE = CACHE_DIR / "research_07_event_metadata.csv"
HMM_FILE = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT_TRADES = RESULTS_DIR / "research_08ab_local_candidate_trades.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08ab_local_candidate_summary.csv"


# =============================================================================
# CANDIDATES FROM 08Z
# =============================================================================

CANDIDATES = [
    {
        "strategy_name": "MRS2",
        "candidate_id": "MRS2_NEW",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_low": 80,
        "vol_high": 100,
        "zscore": 2.0,
        "tp": 27.5,
        "sl": 25.0,
        "horizon": 30,
    },
    {
        "strategy_name": "MRL1",
        "candidate_id": "MRL1_NEW",
        "side": "LONG",
        "hmm_state": 1,
        "vol_low": 20,
        "vol_high": 40,
        "zscore": 2.5,
        "tp": 25.0,
        "sl": 37.5,
        "horizon": 8,
    },
    {
        "strategy_name": "MRL2",
        "candidate_id": "MRL2_NEW",
        "side": "LONG",
        "hmm_state": 2,
        "vol_low": 60,
        "vol_high": 80,
        "zscore": 3.5,
        "tp": 35.0,
        "sl": 32.5,
        "horizon": 3,
    },
]


# =============================================================================
# CAUSAL PERCENTILE
# =============================================================================


class FenwickTree:
    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.tree = np.zeros(
            size + 1,
            dtype=np.int64,
        )

    def update(
        self,
        index: int,
        value: int = 1,
    ):

        i = index + 1

        while i <= self.size:
            self.tree[i] += value
            i += i & -i

    def query(
        self,
        index: int,
    ) -> int:

        if index < 0:
            return 0

        i = min(
            index + 1,
            self.size,
        )

        total = 0

        while i > 0:
            total += self.tree[i]
            i -= i & -i

        return int(total)


def causal_percentile(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    result = np.full(
        n,
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(values)

    if not finite.any():
        return result

    unique_values = np.unique(values[finite])

    coordinate = {value: i for i, value in enumerate(unique_values)}

    tree = FenwickTree(len(unique_values))

    count = 0

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        idx = coordinate[value]

        less_equal = tree.query(idx)

        result[i] = less_equal / count if count > 0 else np.nan

        tree.update(idx)

        count += 1

    return result


# =============================================================================
# VOLATILITY BUCKETS
# =============================================================================


def add_volatility_bucket(
    rth: pd.DataFrame,
) -> pd.DataFrame:

    result = rth.copy()

    if "realized_vol_30" not in result.columns:
        close = result["close"]

        returns = close.pct_change()

        result["realized_vol_30"] = returns.rolling(30).std()

    percentile = causal_percentile(result["realized_vol_30"].to_numpy(dtype=float))

    result["vol_percentile"] = percentile * 100.0

    bins = [
        0,
        20,
        40,
        60,
        80,
        100.000001,
    ]

    labels = [
        "VOL0-20",
        "VOL20-40",
        "VOL40-60",
        "VOL60-80",
        "VOL80-100",
    ]

    result["vol_bucket"] = pd.cut(
        result["vol_percentile"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return result


# =============================================================================
# FUTURE OHLC
# =============================================================================


def build_future_ohlc(
    rth: pd.DataFrame,
    max_horizon: int,
):
    """

    Build future OHLC arrays.

    Shape:

        events x horizon

    Offset 1 means the next RTH bar.
    """

    high = rth["high"].to_numpy(dtype=float)

    low = rth["low"].to_numpy(dtype=float)

    close = rth["close"].to_numpy(dtype=float)

    n = len(rth)

    future_high = np.full(
        (n, max_horizon),
        np.nan,
        dtype=float,
    )

    future_low = np.full(
        (n, max_horizon),
        np.nan,
        dtype=float,
    )

    future_close = np.full(
        (n, max_horizon),
        np.nan,
        dtype=float,
    )

    for offset in range(
        1,
        max_horizon + 1,
    ):
        destination = offset - 1

        future_high[
            :-offset,
            destination,
        ] = high[offset:]

        future_low[
            :-offset,
            destination,
        ] = low[offset:]

        future_close[
            :-offset,
            destination,
        ] = close[offset:]

    return (
        future_high,
        future_low,
        future_close,
    )


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(
    r: np.ndarray,
) -> dict:

    r = np.asarray(
        r,
        dtype=float,
    )

    r = r[np.isfinite(r)]

    if len(r) == 0:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "net_r": np.nan,
            "expectancy_r": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_dd_r": np.nan,
        }

    wins = r[r > 0]

    losses = r[r < 0]

    gross_profit = wins.sum()

    gross_loss = abs(losses.sum())

    equity = np.cumsum(r)

    running_max = np.maximum.accumulate(equity)

    drawdown = equity - running_max

    return {
        "n": len(r),
        "wins": len(wins),
        "losses": len(losses),
        "net_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": (
            float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        ),
        "max_dd_r": float(drawdown.min()),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — 08AB LOCAL CANDIDATE CONFIRMATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # LOAD EXACT CANONICAL PIPELINE
    # -------------------------------------------------------------------------

    print()
    print("Loading canonical data through load_data()...")

    market = load_data()

    print(f"Market rows: {len(market):,}")

    if "market_period" not in market.columns:
        raise RuntimeError("market_period missing from canonical data.")

    # -------------------------------------------------------------------------
    # EXACT RESEARCH 07 RTH
    # -------------------------------------------------------------------------

    rth = market.loc[market["market_period"].eq("RTH")].copy()

    rth = rth.sort_values("timestamp ET")

    rth = rth.reset_index(drop=True)

    print(f"RTH rows: {len(rth):,}")

    if len(rth) != 825_746:
        raise RuntimeError(f"RTH mismatch. Expected 825,746, got {len(rth):,}.")

    # -------------------------------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------------------------------

    print()
    print("Building causal volatility buckets...")

    rth = add_volatility_bucket(rth)

    print(
        rth["vol_bucket"].value_counts(
            sort=False,
            dropna=False,
        )
    )

    # -------------------------------------------------------------------------
    # EVENT METADATA
    # -------------------------------------------------------------------------

    events = pd.read_csv(EVENT_FILE)

    print()
    print(f"Research 07 events: {len(events):,}")

    if not np.array_equal(
        events["event_id"].to_numpy(),
        np.arange(
            len(events),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError("event_id is not identical to event row index.")

    data_indices = events["data_index"].to_numpy(dtype=np.int64)

    if not ((data_indices >= 0) & (data_indices < len(rth))).all():
        raise RuntimeError("Invalid Research 07 data_index.")

    # -------------------------------------------------------------------------
    # HMM
    # -------------------------------------------------------------------------

    hmm = pd.read_csv(HMM_FILE)

    if len(hmm) != len(events):
        raise RuntimeError("HMM/event dimensions do not match.")

    if "hmm_state" in hmm.columns:
        hmm_states = hmm["hmm_state"].to_numpy()

    elif "state" in hmm.columns:
        hmm_states = hmm["state"].to_numpy()

    else:
        raise RuntimeError("HMM state column not found.")

    # -------------------------------------------------------------------------
    # MAP EVENT CONTEXT
    # -------------------------------------------------------------------------

    event_context = rth.iloc[data_indices].copy()

    event_context = event_context.reset_index(drop=True)

    if not np.allclose(
        event_context["close"].to_numpy(),
        events["close"].to_numpy(),
        equal_nan=True,
    ):
        raise RuntimeError("Event close prices do not match RTH data.")

    # -------------------------------------------------------------------------
    # FUTURE PATHS
    # -------------------------------------------------------------------------

    max_horizon = max(candidate["horizon"] for candidate in CANDIDATES)

    print()
    print(f"Building future OHLC paths: H={max_horizon}")

    (
        future_high_all,
        future_low_all,
        future_close_all,
    ) = build_future_ohlc(
        rth,
        max_horizon,
    )

    # Convert from RTH-row indexing to event indexing.

    future_high = future_high_all[data_indices]

    future_low = future_low_all[data_indices]

    future_close = future_close_all[data_indices]

    print(
        "Future OHLC shape:",
        future_close.shape,
    )

    # -------------------------------------------------------------------------
    # EVALUATE CANDIDATES
    # -------------------------------------------------------------------------

    all_trades = []
    summaries = []

    zscores = events["zscore_30"].to_numpy(dtype=float)

    vol_buckets = event_context["vol_bucket"].astype(str).to_numpy()

    timestamps = events["timestamp"].to_numpy()

    entries = events["close"].to_numpy(dtype=float)

    windows = events["window"].to_numpy()

    for candidate in CANDIDATES:
        print()
        print("-" * 100)
        print(f"{candidate['strategy_name']} / {candidate['candidate_id']}")
        print("-" * 100)

        side = candidate["side"]

        tp = float(candidate["tp"])

        sl = float(candidate["sl"])

        horizon = int(candidate["horizon"])

        expected_bucket = f"VOL{candidate['vol_low']}-{candidate['vol_high']}"

        # -------------------------------------------------------------
        # SIGNAL
        # -------------------------------------------------------------

        if side == "LONG":
            signal = zscores <= -candidate["zscore"]

        else:
            signal = zscores >= candidate["zscore"]

        signal &= hmm_states == candidate["hmm_state"]

        signal &= vol_buckets == expected_bucket

        event_ids = np.flatnonzero(signal)

        print(f"Selected events: {len(event_ids):,}")

        if len(event_ids) == 0:
            raise RuntimeError("Candidate produced zero events.")

        # -------------------------------------------------------------
        # PATHS
        # -------------------------------------------------------------

        fh = future_high[
            event_ids,
            :horizon,
        ]

        fl = future_low[
            event_ids,
            :horizon,
        ]

        fc = future_close[
            event_ids,
            :horizon,
        ]

        entry = entries[event_ids]

        # -------------------------------------------------------------
        # RESULT ARRAYS
        # -------------------------------------------------------------

        results = np.full(
            len(event_ids),
            "TIMEOUT",
            dtype=object,
        )

        r_values = np.zeros(
            len(event_ids),
            dtype=float,
        )

        bars = np.full(
            len(event_ids),
            horizon,
            dtype=int,
        )

        # -------------------------------------------------------------
        # TRADE EVALUATION
        # -------------------------------------------------------------

        for j in range(len(event_ids)):
            if side == "LONG":
                favorable = fh[j] - entry[j]

                adverse = entry[j] - fl[j]

            else:
                favorable = entry[j] - fl[j]

                adverse = fh[j] - entry[j]

            target_hits = np.flatnonzero(favorable >= tp)

            stop_hits = np.flatnonzero(adverse >= sl)

            target_idx = target_hits[0] if len(target_hits) else None

            stop_idx = stop_hits[0] if len(stop_hits) else None

            # -----------------------------------------------------
            # BOTH HIT
            # -----------------------------------------------------

            if target_idx is not None and stop_idx is not None:
                if target_idx < stop_idx:
                    results[j] = "WIN"

                    r_values[j] = tp / sl

                    bars[j] = target_idx + 1

                elif stop_idx < target_idx:
                    results[j] = "LOSS"

                    r_values[j] = -1.0

                    bars[j] = stop_idx + 1

                else:
                    # Same bar -> STOP FIRST.

                    results[j] = "LOSS"

                    r_values[j] = -1.0

                    bars[j] = stop_idx + 1

            # -----------------------------------------------------
            # TARGET ONLY
            # -----------------------------------------------------

            elif target_idx is not None:
                results[j] = "WIN"

                r_values[j] = tp / sl

                bars[j] = target_idx + 1

            # -----------------------------------------------------
            # STOP ONLY
            # -----------------------------------------------------

            elif stop_idx is not None:
                results[j] = "LOSS"

                r_values[j] = -1.0

                bars[j] = stop_idx + 1

            # -----------------------------------------------------
            # TIMEOUT
            # -----------------------------------------------------

            else:
                final_close = fc[
                    j,
                    -1,
                ]

                if side == "LONG":
                    pnl = final_close - entry[j]

                else:
                    pnl = entry[j] - final_close

                r_values[j] = pnl / sl

                results[j] = "TIMEOUT"

        # ---------------------------------------------------------------------
        # TRADES
        # ---------------------------------------------------------------------

        for j, event_id in enumerate(event_ids):
            all_trades.append(
                {
                    "strategy_name": candidate["strategy_name"],
                    "candidate_id": candidate["candidate_id"],
                    "side": side,
                    "hmm_state": candidate["hmm_state"],
                    "vol_bucket": expected_bucket,
                    "zscore": candidate["zscore"],
                    "tp": tp,
                    "sl": sl,
                    "rr": tp / sl,
                    "horizon": horizon,
                    "event_id": int(event_id),
                    "window": windows[event_id],
                    "timestamp": timestamps[event_id],
                    "entry": entries[event_id],
                    "result": results[j],
                    "r": r_values[j],
                    "bars_to_result": bars[j],
                }
            )

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        m = calculate_metrics(r_values)

        summaries.append(
            {
                **candidate,
                **m,
            }
        )

        print(
            f"N={m['n']:,} | "
            f"Net={m['net_r']:.2f}R | "
            f"Exp={m['expectancy_r']:.6f}R | "
            f"PF={m['profit_factor']:.4f} | "
            f"WR={m['win_rate']:.2%} | "
            f"DD={m['max_dd_r']:.2f}R"
        )

        print(pd.Series(results).value_counts().to_string())

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    trades = pd.DataFrame(all_trades)

    summary = pd.DataFrame(summaries)

    trades.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()
    print("=" * 100)
    print("08AB COMPLETE")
    print("=" * 100)

    print(f"Trades saved to:\n{OUTPUT_TRADES}")

    print(f"Summary saved to:\n{OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
