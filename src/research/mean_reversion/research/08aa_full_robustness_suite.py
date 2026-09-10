from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

EVENTS_FILE = CACHE_DIR / "research_07_event_metadata.csv"
HMM_FILE = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"


# ============================================================
# CANDIDATES
# ============================================================
# IMPORTANT:
# These are NEW parameters discovered in 08Y / 08Z.
# The old 5/2 candidates are intentionally NOT included.
#
# We are validating this new branch only.
# ============================================================

CANDIDATES = [
    {
        "candidate_id": "MRS2_NEW",
        "strategy_name": "MRS2",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "VOL80-100",
        "zscore_threshold": 2.0,
        "tp": 27.5,
        "sl": 25.0,
        "horizon": 30,
    },
    {
        "candidate_id": "MRL1_NEW",
        "strategy_name": "MRL1",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "VOL20-40",
        "zscore_threshold": 2.5,
        "tp": 25.0,
        "sl": 37.5,
        "horizon": 8,
    },
]


# ============================================================
# RESEARCH 07 LOADER
# ============================================================


def load_research_07_module():
    """
    Load Research 07 directly so that we can reuse its exact
    RTH preparation logic instead of recreating it.
    """
    path = BASE_DIR / "research" / "07_path_dependent_long_short.py"

    spec = importlib.util.spec_from_file_location(
        "research_07_path_dependent_long_short",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Research 07: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def prepare_rth_exactly_like_research_07(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    IMPORTANT:
    Reuse Research 07's actual prepare_rth implementation.
    """
    research_07 = load_research_07_module()

    if not hasattr(research_07, "prepare_rth"):
        raise RuntimeError("Research 07 does not expose prepare_rth().")

    return research_07.prepare_rth(market.copy())


# ============================================================
# HELPERS
# ============================================================


def normalize_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    )


def calculate_metrics(results: pd.DataFrame) -> dict:
    if results.empty:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "win_rate": np.nan,
            "net_r": np.nan,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": np.nan,
        }

    r = results["r"].to_numpy(dtype=float)

    wins = int(np.sum(r > 0))
    losses = int(np.sum(r < 0))
    unresolved = int(np.sum(r == 0))

    gross_profit = float(r[r > 0].sum()) if wins else 0.0
    gross_loss = float(-r[r < 0].sum()) if losses else 0.0

    equity = np.cumsum(r)
    running_max = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]

    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "n": len(results),
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "win_rate": wins / len(results),
        "net_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "profit_factor": (gross_profit / gross_loss if gross_loss > 0 else np.inf),
        "max_drawdown_r": max_dd,
    }


# ============================================================
# EVENT / HMM ALIGNMENT
# ============================================================


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_FILE)

    events["timestamp"] = normalize_timestamp(events["timestamp"])

    events = events.sort_values("event_id").reset_index(drop=True)

    expected = np.arange(len(events))

    if not np.array_equal(
        events["event_id"].to_numpy(),
        expected,
    ):
        raise RuntimeError("Research 07 event_id is not contiguous 0..N-1.")

    return events


def load_hmm() -> pd.DataFrame:
    hmm = pd.read_csv(HMM_FILE)

    if "timestamp" in hmm.columns:
        hmm["timestamp"] = normalize_timestamp(hmm["timestamp"])

    return hmm.reset_index(drop=True)


# ============================================================
# VOLATILITY CONTEXT
# ============================================================


def add_causal_volatility_bucket(
    rth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add causal expanding volatility percentile using a Fenwick Tree.

    IMPORTANT:
    The percentile at row i only uses volatility observations
    strictly before row i.

    This is the same efficient ranking approach used in the
    previous Mean Reversion research and avoids O(N^2) sorting.
    """

    rth = rth.copy()

    if "realized_vol_30" not in rth.columns:
        raise RuntimeError("RTH dataset does not contain realized_vol_30.")

    values = rth["realized_vol_30"].to_numpy(dtype=float)

    n = len(values)

    percentile = np.full(n, np.nan, dtype=float)

    finite_mask = np.isfinite(values)

    finite_values = values[finite_mask]

    if len(finite_values) == 0:
        rth["causal_vol_percentile"] = percentile
        rth["vol_bucket_id"] = -1
        rth["vol_bucket"] = "UNKNOWN"
        return rth

    # --------------------------------------------------------
    # Coordinate compression
    # --------------------------------------------------------

    unique_values = np.unique(finite_values)

    value_to_index = {float(value): i for i, value in enumerate(unique_values)}

    size = len(unique_values)

    # Fenwick tree
    tree = np.zeros(
        size + 1,
        dtype=np.int64,
    )

    def update(index: int) -> None:
        """
        Add one observation at zero-based compressed index.
        """
        i = index + 1

        while i <= size:
            tree[i] += 1
            i += i & -i

    def query(index: int) -> int:
        """
        Number of observations with compressed index <= index.
        """
        if index < 0:
            return 0

        i = min(index + 1, size)

        total = 0

        while i > 0:
            total += tree[i]
            i -= i & -i

        return int(total)

    # --------------------------------------------------------
    # Causal expanding percentile
    # --------------------------------------------------------

    observations_seen = 0

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        compressed_index = value_to_index[float(value)]

        # IMPORTANT:
        # Query BEFORE inserting current observation.
        #
        # Therefore the percentile is causal and uses only
        # information available before this event.
        if observations_seen > 0:
            percentile[i] = query(compressed_index) / observations_seen

        update(compressed_index)

        observations_seen += 1

    # --------------------------------------------------------
    # Volatility buckets
    # --------------------------------------------------------

    buckets = np.full(
        n,
        -1,
        dtype=np.int8,
    )

    valid = np.isfinite(percentile)

    buckets[valid] = np.minimum(
        (percentile[valid] * 5).astype(np.int8),
        4,
    )

    rth["causal_vol_percentile"] = percentile

    rth["vol_bucket_id"] = buckets

    rth["vol_bucket"] = np.select(
        [
            buckets == 0,
            buckets == 1,
            buckets == 2,
            buckets == 3,
            buckets == 4,
        ],
        [
            "VOL0-20",
            "VOL20-40",
            "VOL40-60",
            "VOL60-80",
            "VOL80-100",
        ],
        default="UNKNOWN",
    )

    return rth


# ============================================================
# FUTURE OHLC
# ============================================================


def build_future_ohlc(
    rth: pd.DataFrame,
    max_horizon: int,
):
    high = rth["high"].to_numpy(dtype=float)
    low = rth["low"].to_numpy(dtype=float)
    close = rth["close"].to_numpy(dtype=float)

    n = len(rth)

    future_high = np.full(
        (n, max_horizon),
        np.nan,
    )

    future_low = np.full(
        (n, max_horizon),
        np.nan,
    )

    future_close = np.full(
        (n, max_horizon),
        np.nan,
    )

    for offset in range(1, max_horizon + 1):
        if offset >= n:
            break

        future_high[:-offset, offset - 1] = high[offset:]
        future_low[:-offset, offset - 1] = low[offset:]
        future_close[:-offset, offset - 1] = close[offset:]

    return future_high, future_low, future_close


# ============================================================
# INTRABAR EVALUATION
# ============================================================


def evaluate_trade(
    entry: float,
    side: str,
    tp: float,
    sl: float,
    horizon: int,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    event_row: int,
) -> tuple[str, float, int]:

    for bar in range(horizon):
        fh = future_high[event_row, bar]
        fl = future_low[event_row, bar]
        fc = future_close[event_row, bar]

        if not (np.isfinite(fh) and np.isfinite(fl) and np.isfinite(fc)):
            continue

        if side == "LONG":
            tp_hit = fh >= entry + tp
            sl_hit = fl <= entry - sl

            # Conservative same-bar resolution:
            # STOP FIRST
            if tp_hit and sl_hit:
                return "LOSS", -1.0, bar + 1

            if sl_hit:
                return "LOSS", -1.0, bar + 1

            if tp_hit:
                return "WIN", tp / sl, bar + 1

        else:
            tp_hit = fl <= entry - tp
            sl_hit = fh >= entry + sl

            if tp_hit and sl_hit:
                return "LOSS", -1.0, bar + 1

            if sl_hit:
                return "LOSS", -1.0, bar + 1

            if tp_hit:
                return "WIN", tp / sl, bar + 1

    # Timeout: use final horizon close
    final_close = future_close[
        event_row,
        horizon - 1,
    ]

    if not np.isfinite(final_close):
        return "UNRESOLVED", 0.0, horizon

    if side == "LONG":
        pnl = final_close - entry
    else:
        pnl = entry - final_close

    if pnl > 0:
        return "TIMEOUT_WIN", pnl / sl, horizon

    if pnl < 0:
        return "TIMEOUT_LOSS", pnl / sl, horizon

    return "UNRESOLVED", 0.0, horizon


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08AA — NEW PARAMETER BRANCH FULL ROBUSTNESS")
    print("=" * 72)

    print("\n[1/8] Loading Research 07 events...")
    events = load_events()

    print(f"Events: {len(events):,}")

    print("\n[2/8] Loading causal HMM...")
    hmm = load_hmm()

    print(f"HMM rows: {len(hmm):,}")

    if len(hmm) != len(events):
        raise RuntimeError("HMM rows do not match Research 07 events.")

    print("\n[3/8] Loading market data...")
    market = load_data()

    print(f"Market rows: {len(market):,}")

    rth = prepare_rth_exactly_like_research_07(market)

    print(f"RTH rows: {len(rth):,}")

    if len(rth) != 825_746:
        raise RuntimeError(
            f"RTH row count mismatch. "
            f"Expected 825,746, got {len(rth):,}. "
            f"STOPPING instead of inventing a different RTH definition."
        )

    rth = rth.reset_index(drop=True)

    print("RTH row count: PASS")

    # Research 07 data_index refers to this exact RTH dataframe.
    valid_indices = events["data_index"].to_numpy(dtype=int)

    if valid_indices.min() < 0:
        raise RuntimeError("Negative data_index found.")

    if valid_indices.max() >= len(rth):
        raise RuntimeError(
            "Research 07 data_index cannot be mapped into the reproduced RTH dataframe."
        )

    print("data_index → RTH mapping: PASS")

    print("\n[5/8] Building causal volatility context...")

    rth = add_causal_volatility_bucket(rth)

    event_market = rth.iloc[valid_indices].reset_index(drop=True)

    if len(event_market) != len(events):
        raise RuntimeError("Event/RTH mapping changed row count.")

    print("Event context mapping: PASS")

    print("\n[6/8] Building future OHLC arrays...")

    max_horizon = max(candidate["horizon"] for candidate in CANDIDATES)

    future_high, future_low, future_close = build_future_ohlc(
        rth,
        max_horizon,
    )

    print(f"Future OHLC shape: {future_high.shape}")

    print("\n[7/8] Evaluating NEW candidates...")

    all_results = []

    for candidate in CANDIDATES:
        print(
            f"\n  {candidate['candidate_id']}: "
            f"TP={candidate['tp']} "
            f"SL={candidate['sl']} "
            f"H={candidate['horizon']}"
        )

        mask = (hmm["hmm_state"] == candidate["hmm_state"]) & (
            event_market["vol_bucket"] == candidate["vol_bucket"]
        )

        if candidate["side"] == "LONG":
            mask &= events["zscore_30"] <= -candidate["zscore_threshold"]
        else:
            mask &= events["zscore_30"] >= candidate["zscore_threshold"]

        selected = np.flatnonzero(mask.to_numpy())

        print(f"  Selected events: {len(selected):,}")

        rows = []

        for event_id in selected:
            entry = float(events.iloc[event_id]["close"])

            result, r_value, bars = evaluate_trade(
                entry=entry,
                side=candidate["side"],
                tp=candidate["tp"],
                sl=candidate["sl"],
                horizon=candidate["horizon"],
                future_high=future_high,
                future_low=future_low,
                future_close=future_close,
                event_row=int(events.iloc[event_id]["data_index"]),
            )

            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "strategy_name": candidate["strategy_name"],
                    "event_id": int(event_id),
                    "timestamp": events.iloc[event_id]["timestamp"],
                    "window": events.iloc[event_id]["window"],
                    "side": candidate["side"],
                    "tp": candidate["tp"],
                    "sl": candidate["sl"],
                    "rr": candidate["tp"] / candidate["sl"],
                    "horizon": candidate["horizon"],
                    "result": result,
                    "r": r_value,
                    "bars": bars,
                }
            )

        result_df = pd.DataFrame(rows)

        metrics = calculate_metrics(result_df)

        print(
            f"  N={metrics['n']:,} "
            f"WR={metrics['win_rate']:.4f} "
            f"Exp={metrics['expectancy_r']:.6f} "
            f"PF={metrics['profit_factor']:.4f} "
            f"Net={metrics['net_r']:.2f}R "
            f"DD={metrics['max_drawdown_r']:.2f}R"
        )

        all_results.append(result_df)

    trades = pd.concat(
        all_results,
        ignore_index=True,
    )

    print("\n[8/8] Temporal robustness...")

    summary_rows = []

    for candidate in CANDIDATES:
        candidate_trades = trades[
            trades["candidate_id"] == candidate["candidate_id"]
        ].copy()

        windows = sorted(candidate_trades["window"].dropna().unique())

        positive_windows = 0

        for window in windows:
            w = candidate_trades[candidate_trades["window"] == window]

            metrics = calculate_metrics(w)

            if metrics["net_r"] > 0:
                positive_windows += 1

            summary_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "window": window,
                    **metrics,
                }
            )

        print(
            f"  {candidate['candidate_id']}: "
            f"{positive_windows}/{len(windows)} "
            f"positive windows"
        )

    window_results = pd.DataFrame(summary_rows)

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    window_file = RESULTS_DIR / "research_08aa_window_results.csv"

    window_results.to_csv(
        window_file,
        index=False,
    )

    print("\n" + "=" * 72)
    print("08AA COMPLETE")
    print("=" * 72)

    print(f"Trades: {OUTPUT_FILE}")
    print(f"Windows: {window_file}")

    print("\nIMPORTANT: These results validate the NEW parameter branch only.")


if __name__ == "__main__":
    main()
