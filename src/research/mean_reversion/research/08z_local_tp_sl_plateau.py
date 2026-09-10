from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

RESEARCH_07_EVENTS = CACHE_DIR / "research_07_event_metadata.csv"

RESEARCH_08B_HMM = CACHE_DIR / "research_08b_causal_hmm_states.csv"


# ============================================================
# LOCAL SEARCH GRID
# ============================================================
#
# 08Y was the global map.
#
# 08Z is deliberately LOCAL and finer.
#
# TP and SL are COMPLETELY INDEPENDENT.
# No RR restriction is imposed.
#
# The grid is centered around the promising regions found
# in 08Y, with enough surrounding space to detect whether
# there is a genuine plateau rather than a single peak.
#
# ============================================================

SEARCH_GRIDS = {
    "MRS2": {
        "tp": [
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
        ],
        "sl": [
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
        ],
        "horizon": [
            12,
            20,
            30,
        ],
    },
    "MRL1": {
        "tp": [
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
            37.5,
            40.0,
            42.5,
        ],
        "sl": [
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
            37.5,
            40.0,
            42.5,
        ],
        "horizon": [
            8,
            12,
            20,
            30,
        ],
    },
    "MRL2": {
        "tp": [
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
            37.5,
            40.0,
            42.5,
        ],
        "sl": [
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
            37.5,
            40.0,
            42.5,
        ],
        "horizon": [
            2,
            3,
            5,
            8,
        ],
    },
}


# ============================================================
# STRATEGY CONTEXTS
# ============================================================

CONTEXTS = {
    "MRS2": {
        "candidate_id": "C01",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": 4,
        "vol_label": "VOL80-100",
        "z_threshold": 2.0,
    },
    "MRL1": {
        "candidate_id": "C02",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": 1,
        "vol_label": "VOL20-40",
        "z_threshold": 2.5,
    },
    "MRL2": {
        "candidate_id": "C06",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": 3,
        "vol_label": "VOL60-80",
        "z_threshold": 3.5,
    },
}


# ============================================================
# HELPERS
# ============================================================


def ensure_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def classify_vol_bucket(percentile: float) -> int:

    if not np.isfinite(percentile):
        return -1

    if percentile < 20.0:
        return 0

    if percentile < 40.0:
        return 1

    if percentile < 60.0:
        return 2

    if percentile < 80.0:
        return 3

    return 4


# ============================================================
# CAUSAL PERCENTILE
# ============================================================


def causal_percentile(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    result = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    finite_mask = np.isfinite(values)

    if not finite_mask.any():
        return result

    finite_values = values[finite_mask]

    unique_values = np.unique(finite_values)

    size = len(unique_values)

    tree = np.zeros(
        size + 1,
        dtype=np.int64,
    )

    coordinate = {value: idx for idx, value in enumerate(unique_values)}

    def update(index: int) -> None:

        i = index + 1

        while i <= size:
            tree[i] += 1

            i += i & -i

    def query(index: int) -> int:

        if index < 0:
            return 0

        i = min(
            index + 1,
            size,
        )

        total = 0

        while i > 0:
            total += tree[i]

            i -= i & -i

        return int(total)

    seen = 0

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        idx = coordinate[value]

        update(idx)

        seen += 1

        count_leq = query(idx)

        result[i] = 100.0 * count_leq / seen

    return result


# ============================================================
# RESEARCH 07 RTH
# ============================================================


def prepare_rth(
    data: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "timestamp ET",
        "market_period",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing RTH columns: {sorted(missing)}")

    rth = data.loc[data["market_period"].astype(str).eq("RTH")].copy()

    rth = rth.sort_values("timestamp ET").reset_index(drop=True)

    return rth


# ============================================================
# EVENT MAPPING
# ============================================================


def validate_event_mapping(
    events: pd.DataFrame,
    rth: pd.DataFrame,
) -> None:

    event_ids = events["event_id"].to_numpy(dtype=np.int64)

    data_indices = events["data_index"].to_numpy(dtype=np.int64)

    expected = np.arange(
        len(events),
        dtype=np.int64,
    )

    if not np.array_equal(
        event_ids,
        expected,
    ):
        raise RuntimeError("event_id is not equal to row index.")

    if data_indices.min() < 0:
        raise RuntimeError("Negative data_index detected.")

    if data_indices.max() >= len(rth):
        raise RuntimeError("data_index exceeds RTH length.")

    print(
        f"[08Z] event_id == row index: {np.sum(event_ids == expected)}/{len(event_ids)}"
    )

    print(f"[08Z] data_index range: {data_indices.min()} -> {data_indices.max()}")

    print(f"[08Z] RTH row index range: 0 -> {len(rth) - 1}")

    print("[08Z] exact event -> RTH mapping: PASS")


# ============================================================
# BUILD EVENT CONTEXT
# ============================================================


def build_event_context(
    events: pd.DataFrame,
    hmm_states: pd.DataFrame,
    rth: pd.DataFrame,
) -> pd.DataFrame:

    events = events.copy()

    events["timestamp"] = ensure_utc(events["timestamp"])

    hmm = hmm_states.copy()

    hmm["timestamp"] = ensure_utc(hmm["timestamp"])

    # --------------------------------------------------------
    # Volatility column
    # --------------------------------------------------------

    volatility_column = None

    for candidate in [
        "realized_vol_30",
        "realized_vol",
        "volatility",
    ]:
        if candidate in rth.columns:
            volatility_column = candidate
            break

    if volatility_column is None:
        raise ValueError("Could not find realized volatility column in RTH data.")

    volatility = rth[volatility_column].to_numpy(dtype=float)

    print(
        "[08Z] Calculating causal volatility "
        f"percentile over {len(volatility):,} RTH rows..."
    )

    percentile = causal_percentile(volatility)

    data_indices = events["data_index"].to_numpy(dtype=np.int64)

    events["realized_vol_30"] = volatility[data_indices]

    events["vol_percentile"] = percentile[data_indices]

    events["vol_bucket"] = (
        events["vol_percentile"].map(classify_vol_bucket).astype(np.int16)
    )

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    if "event_id" in hmm.columns:
        hmm_small = hmm[
            [
                "event_id",
                "hmm_state",
            ]
        ].copy()

        events = events.merge(
            hmm_small,
            on="event_id",
            how="left",
            validate="one_to_one",
        )

    elif len(hmm) == len(events):
        events["hmm_state"] = hmm["hmm_state"].to_numpy(dtype=np.int16)

    else:
        raise RuntimeError("Cannot align HMM states.")

    if events["hmm_state"].isna().any():
        raise RuntimeError("Missing HMM states detected.")

    if events["vol_bucket"].eq(-1).any():
        raise RuntimeError("Missing volatility buckets detected.")

    return events


# ============================================================
# SELECT STRATEGY EVENTS
# ============================================================


def select_context_events(
    events: pd.DataFrame,
    context: Dict,
) -> pd.DataFrame:

    mask = events["hmm_state"].astype(int).eq(context["hmm_state"])

    mask &= events["vol_bucket"].astype(int).eq(context["vol_bucket"])

    if context["side"] == "LONG":
        mask &= events["zscore_30"] <= -context["z_threshold"]

    elif context["side"] == "SHORT":
        mask &= events["zscore_30"] >= context["z_threshold"]

    else:
        raise ValueError(f"Unknown side: {context['side']}")

    return events.loc[mask].sort_values("event_id").reset_index(drop=True)


# ============================================================
# FUTURE ARRAYS
# ============================================================


def build_future_arrays(
    rth: pd.DataFrame,
    data_indices: np.ndarray,
    max_horizon: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    highs = rth["high"].to_numpy(dtype=float)

    lows = rth["low"].to_numpy(dtype=float)

    closes = rth["close"].to_numpy(dtype=float)

    offsets = np.arange(
        1,
        max_horizon + 1,
        dtype=np.int64,
    )

    indices = data_indices[:, None] + offsets[None, :]

    valid = indices < len(rth)

    safe_indices = np.minimum(
        indices,
        len(rth) - 1,
    )

    future_high = highs[safe_indices].copy()

    future_low = lows[safe_indices].copy()

    future_close = closes[safe_indices].copy()

    future_high[~valid] = np.nan
    future_low[~valid] = np.nan
    future_close[~valid] = np.nan

    return (
        future_high,
        future_low,
        future_close,
    )


# ============================================================
# INTRABAR EVALUATION
# ============================================================


def evaluate_intrabar_batch(
    entries: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    side: str,
    tp: float,
    sl: float,
    horizon: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    n = len(entries)

    result = np.zeros(
        n,
        dtype=np.int8,
    )

    r_values = np.zeros(
        n,
        dtype=float,
    )

    bars = np.full(
        n,
        horizon,
        dtype=np.int16,
    )

    highs = future_high[:, :horizon]

    lows = future_low[:, :horizon]

    closes = future_close[:, :horizon]

    if side == "LONG":
        favorable = highs - entries[:, None]

        adverse = entries[:, None] - lows

    elif side == "SHORT":
        favorable = entries[:, None] - lows

        adverse = highs - entries[:, None]

    else:
        raise ValueError(f"Unknown side: {side}")

    target_hit = (favorable >= tp) & np.isfinite(favorable)

    stop_hit = (adverse >= sl) & np.isfinite(adverse)

    any_target = target_hit.any(axis=1)

    any_stop = stop_hit.any(axis=1)

    first_target = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
    )

    first_stop = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
    )

    target_rows = np.where(any_target)[0]

    if len(target_rows):
        first_target[target_rows] = (
            np.argmax(
                target_hit[target_rows],
                axis=1,
            )
            + 1
        )

    stop_rows = np.where(any_stop)[0]

    if len(stop_rows):
        first_stop[stop_rows] = (
            np.argmax(
                stop_hit[stop_rows],
                axis=1,
            )
            + 1
        )

    # --------------------------------------------------------
    # STOP FIRST on same bar.
    # --------------------------------------------------------

    win = any_target & (first_target < first_stop)

    loss = any_stop & (first_stop <= first_target)

    result[win] = 1
    result[loss] = -1

    r_values[win] = tp / sl

    r_values[loss] = -1.0

    barrier = win | loss

    barrier_rows = np.where(barrier)[0]

    if len(barrier_rows):
        bars[barrier_rows] = np.where(
            win[barrier_rows],
            first_target[barrier_rows],
            first_stop[barrier_rows],
        )

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    timeout = ~barrier

    timeout_rows = np.where(timeout)[0]

    if len(timeout_rows):
        final_close = closes[
            timeout_rows,
            horizon - 1,
        ]

        finite = np.isfinite(final_close)

        valid_rows = timeout_rows[finite]

        if side == "LONG":
            pnl = final_close[finite] - entries[valid_rows]

        else:
            pnl = entries[valid_rows] - final_close[finite]

        r_values[valid_rows] = pnl / sl

    return (
        result,
        r_values,
        bars,
    )


# ============================================================
# METRICS
# ============================================================


def max_drawdown(
    r_values: np.ndarray,
) -> float:

    if len(r_values) == 0:
        return 0.0

    cumulative = np.cumsum(r_values)

    running_max = np.maximum.accumulate(
        np.concatenate(
            [
                np.array([0.0]),
                cumulative,
            ]
        )
    )[1:]

    drawdown = cumulative - running_max

    return float(drawdown.min())


def profit_factor(
    r_values: np.ndarray,
) -> float:

    gains = r_values[r_values > 0].sum()

    losses = -r_values[r_values < 0].sum()

    if losses <= 0:
        if gains > 0:
            return np.inf

        return np.nan

    return float(gains / losses)


def summarize(
    r_values: np.ndarray,
) -> Dict:

    n = len(r_values)

    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": np.nan,
            "net_r": 0.0,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "max_dd_r": 0.0,
        }

    wins = int(np.sum(r_values > 0))

    losses = int(np.sum(r_values < 0))

    timeouts = int(np.sum(r_values == 0))

    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / n,
        "net_r": float(np.sum(r_values)),
        "expectancy_r": float(np.mean(r_values)),
        "profit_factor": profit_factor(r_values),
        "max_dd_r": max_drawdown(r_values),
    }


# ============================================================
# WINDOW METRICS
# ============================================================


def window_statistics(
    windows: np.ndarray,
    r_values: np.ndarray,
) -> Tuple[
    List[Dict],
    Dict,
]:

    rows = []

    unique_windows = np.sort(np.unique(windows))

    for window in unique_windows:
        mask = windows == window

        stats = summarize(r_values[mask])

        rows.append(
            {
                "window": int(window),
                **stats,
            }
        )

    window_df = pd.DataFrame(rows)

    if window_df.empty:
        aggregate = {
            "windows": 0,
            "positive_windows": 0,
            "positive_window_rate": np.nan,
            "negative_windows": 0,
            "zero_windows": 0,
            "mean_window_expectancy_r": np.nan,
            "median_window_expectancy_r": np.nan,
            "std_window_expectancy_r": np.nan,
            "min_window_net_r": np.nan,
            "max_window_net_r": np.nan,
            "min_window_expectancy_r": np.nan,
            "max_window_expectancy_r": np.nan,
        }

        return rows, aggregate

    positive = int(np.sum(window_df["net_r"] > 0))

    negative = int(np.sum(window_df["net_r"] < 0))

    zero = int(np.sum(window_df["net_r"] == 0))

    aggregate = {
        "windows": len(window_df),
        "positive_windows": positive,
        "positive_window_rate": (positive / len(window_df)),
        "negative_windows": negative,
        "zero_windows": zero,
        "mean_window_expectancy_r": float(window_df["expectancy_r"].mean()),
        "median_window_expectancy_r": float(window_df["expectancy_r"].median()),
        "std_window_expectancy_r": float(window_df["expectancy_r"].std()),
        "min_window_net_r": float(window_df["net_r"].min()),
        "max_window_net_r": float(window_df["net_r"].max()),
        "min_window_expectancy_r": float(window_df["expectancy_r"].min()),
        "max_window_expectancy_r": float(window_df["expectancy_r"].max()),
    }

    return rows, aggregate


# ============================================================
# PLATEAU SCORE
# ============================================================


def calculate_plateau_metrics(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate local-neighborhood robustness.

    A configuration gets its neighborhood defined by the
    immediately adjacent TP and SL values in the corresponding
    strategy grid.

    The goal is NOT to select a single optimum.

    Instead we measure:

        - neighboring configurations
        - positive neighboring configurations
        - neighbor positive rate
        - mean neighboring expectancy
        - minimum neighboring expectancy
        - local robustness

    A strong plateau should have many positive neighbors and
    relatively small degradation around the center.
    """

    rows = []

    for strategy_name, strategy_df in summary_df.groupby("strategy_name"):
        grid = SEARCH_GRIDS[strategy_name]

        tp_values = grid["tp"]
        sl_values = grid["sl"]

        tp_index = {value: idx for idx, value in enumerate(tp_values)}

        sl_index = {value: idx for idx, value in enumerate(sl_values)}

        lookup = {}

        for _, row in strategy_df.iterrows():
            key = (
                float(row["tp"]),
                float(row["sl"]),
                int(row["horizon"]),
            )

            lookup[key] = row

        for _, row in strategy_df.iterrows():
            tp = float(row["tp"])

            sl = float(row["sl"])

            horizon = int(row["horizon"])

            ti = tp_index[tp]
            si = sl_index[sl]

            neighbors = []

            for dt in [-1, 0, 1]:
                for ds in [-1, 0, 1]:
                    if dt == 0 and ds == 0:
                        continue

                    nti = ti + dt
                    nsi = si + ds

                    if (
                        nti < 0
                        or nti >= len(tp_values)
                        or nsi < 0
                        or nsi >= len(sl_values)
                    ):
                        continue

                    ntp = tp_values[nti]

                    nsl = sl_values[nsi]

                    key = (
                        float(ntp),
                        float(nsl),
                        horizon,
                    )

                    if key in lookup:
                        neighbors.append(lookup[key])

            if neighbors:
                neighbor_exp = np.array([float(x["expectancy_r"]) for x in neighbors])

                positive_neighbors = int(np.sum(neighbor_exp > 0))

                neighbor_count = len(neighbor_exp)

                neighbor_positive_rate = positive_neighbors / neighbor_count

                mean_neighbor_exp = float(neighbor_exp.mean())

                min_neighbor_exp = float(neighbor_exp.min())

            else:
                positive_neighbors = 0
                neighbor_count = 0
                neighbor_positive_rate = np.nan
                mean_neighbor_exp = np.nan
                min_neighbor_exp = np.nan

            rows.append(
                {
                    "strategy_name": strategy_name,
                    "tp": tp,
                    "sl": sl,
                    "horizon": horizon,
                    "neighbor_count": neighbor_count,
                    "positive_neighbors": positive_neighbors,
                    "neighbor_positive_rate": (neighbor_positive_rate),
                    "mean_neighbor_expectancy_r": (mean_neighbor_exp),
                    "min_neighbor_expectancy_r": (min_neighbor_exp),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 72)
    print("08Z — LOCAL TP/SL PLATEAU + 22 OOS WINDOWS")
    print("=" * 72)

    print()

    total_configs = 0

    for strategy_name, grid in SEARCH_GRIDS.items():
        count = len(grid["tp"]) * len(grid["sl"]) * len(grid["horizon"])

        total_configs += count

        print(
            f"{strategy_name}: "
            f"{len(grid['tp'])} TP × "
            f"{len(grid['sl'])} SL × "
            f"{len(grid['horizon'])} H "
            f"= {count} configs"
        )

    print(f"Total local configurations: {total_configs}")

    print()

    # ========================================================
    # LOAD EVENTS
    # ========================================================

    print("[08Z] Loading Research 07 events...")

    events = pd.read_csv(RESEARCH_07_EVENTS)

    print(f"[08Z] Research 07 events: {len(events):,}")

    # ========================================================
    # LOAD HMM
    # ========================================================

    print("[08Z] Loading causal HMM states...")

    hmm = pd.read_csv(RESEARCH_08B_HMM)

    print(f"[08Z] HMM rows: {len(hmm):,}")

    # ========================================================
    # MARKET
    # ========================================================

    print("[08Z] Loading market data...")

    market_data = load_data()

    print(f"[08Z] Market rows: {len(market_data):,}")

    # ========================================================
    # RTH
    # ========================================================

    print("[08Z] Preparing exact Research 07 RTH...")

    rth = prepare_rth(market_data)

    print(f"[08Z] RTH rows: {len(rth):,}")

    validate_event_mapping(
        events,
        rth,
    )

    # ========================================================
    # CONTEXT
    # ========================================================

    print("[08Z] Building event context...")

    events = build_event_context(
        events,
        hmm,
        rth,
    )

    # ========================================================
    # FUTURE ARRAYS
    # ========================================================

    max_horizon = max(max(grid["horizon"]) for grid in SEARCH_GRIDS.values())

    data_indices = events["data_index"].to_numpy(dtype=np.int64)

    print(f"[08Z] Building future OHLC arrays to H={max_horizon}...")

    (
        future_high_all,
        future_low_all,
        future_close_all,
    ) = build_future_arrays(
        rth,
        data_indices,
        max_horizon,
    )

    print("[08Z] Future arrays:")

    print(f"    high  {future_high_all.shape}")

    print(f"    low   {future_low_all.shape}")

    print(f"    close {future_close_all.shape}")

    # ========================================================
    # SEARCH
    # ========================================================

    summary_rows = []
    window_rows = []

    completed = 0

    for strategy_name, context in CONTEXTS.items():
        print()
        print("-" * 72)
        print(f"[08Z] CONTEXT: {strategy_name}")
        print("-" * 72)

        selected = select_context_events(
            events,
            context,
        )

        print(f"[08Z] {strategy_name} events: {len(selected):,}")

        if selected.empty:
            continue

        event_ids = selected["event_id"].to_numpy(dtype=np.int64)

        windows = selected["window"].to_numpy(dtype=np.int64)

        entries = selected["close"].to_numpy(dtype=float)

        future_high = future_high_all[event_ids]

        future_low = future_low_all[event_ids]

        future_close = future_close_all[event_ids]

        grid = SEARCH_GRIDS[strategy_name]

        context_total = len(grid["tp"]) * len(grid["sl"]) * len(grid["horizon"])

        context_completed = 0

        for tp in grid["tp"]:
            for sl in grid["sl"]:
                for horizon in grid["horizon"]:
                    (
                        result,
                        r_values,
                        bars,
                    ) = evaluate_intrabar_batch(
                        entries=entries,
                        future_high=future_high,
                        future_low=future_low,
                        future_close=future_close,
                        side=context["side"],
                        tp=tp,
                        sl=sl,
                        horizon=horizon,
                    )

                    global_metrics = summarize(r_values)

                    window_stats, temporal = window_statistics(
                        windows,
                        r_values,
                    )

                    rr = tp / sl

                    summary_rows.append(
                        {
                            "strategy_name": strategy_name,
                            "candidate_id": context["candidate_id"],
                            "side": context["side"],
                            "hmm_state": context["hmm_state"],
                            "vol_bucket": context["vol_bucket"],
                            "vol_label": context["vol_label"],
                            "z_threshold": context["z_threshold"],
                            "tp": tp,
                            "sl": sl,
                            "rr": rr,
                            "horizon": horizon,
                            **global_metrics,
                            **temporal,
                        }
                    )

                    for row in window_stats:
                        window_rows.append(
                            {
                                "strategy_name": strategy_name,
                                "candidate_id": context["candidate_id"],
                                "side": context["side"],
                                "tp": tp,
                                "sl": sl,
                                "rr": rr,
                                "horizon": horizon,
                                **row,
                            }
                        )

                    context_completed += 1
                    completed += 1

                    if (
                        context_completed == 1
                        or context_completed % 50 == 0
                        or context_completed == context_total
                    ):
                        print(
                            f"[08Z] {strategy_name}: "
                            f"{context_completed}/"
                            f"{context_total}"
                        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)

    window_df = pd.DataFrame(window_rows)

    # ========================================================
    # PLATEAU METRICS
    # ========================================================

    print()

    print("[08Z] Calculating local plateau metrics...")

    plateau_df = calculate_plateau_metrics(summary_df)

    summary_df = summary_df.merge(
        plateau_df,
        on=[
            "strategy_name",
            "tp",
            "sl",
            "horizon",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # ROBUSTNESS RANK
    # ========================================================
    #
    # This is NOT a final statistical score.
    #
    # It is simply a convenient ranking emphasizing:
    #
    #   1. OOS positive-window rate
    #   2. positive local neighbors
    #   3. expectancy
    #
    # It is deliberately NOT based only on maximum Net R.
    #
    # ========================================================

    summary_df["plateau_score"] = (
        summary_df["positive_window_rate"] * 0.50
        + summary_df["neighbor_positive_rate"] * 0.25
        + summary_df["expectancy_r"].clip(
            lower=-1,
            upper=1,
        )
        * 0.25
    )

    # ========================================================
    # SORT
    # ========================================================

    summary_df = summary_df.sort_values(
        [
            "strategy_name",
            "plateau_score",
            "positive_window_rate",
            "expectancy_r",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    window_df = window_df.sort_values(
        [
            "strategy_name",
            "tp",
            "sl",
            "horizon",
            "window",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # TOP TEMPORAL ROBUST CONFIGS
    # ========================================================

    robust_df = summary_df.loc[
        (summary_df["positive_window_rate"] >= 0.75)
        & (summary_df["neighbor_positive_rate"] >= 0.50)
        & (summary_df["expectancy_r"] > 0)
    ].copy()

    robust_df = robust_df.sort_values(
        [
            "strategy_name",
            "plateau_score",
            "expectancy_r",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    # ========================================================
    # OUTPUTS
    # ========================================================

    output_summary = RESULTS_DIR / "research_08z_local_plateau_summary.csv"

    output_windows = RESULTS_DIR / "research_08z_local_plateau_windows.csv"

    output_robust = RESULTS_DIR / "research_08z_robust_configs.csv"

    summary_df.to_csv(
        output_summary,
        index=False,
    )

    window_df.to_csv(
        output_windows,
        index=False,
    )

    robust_df.to_csv(
        output_robust,
        index=False,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("08Z COMPLETE")
    print("=" * 72)

    print(f"Completed configurations: {completed}")

    print()

    for strategy_name in CONTEXTS:
        strategy_results = summary_df.loc[summary_df["strategy_name"] == strategy_name]

        if strategy_results.empty:
            continue

        print("-" * 72)

        print(f"TOP TEMPORALLY ROBUST — {strategy_name}")

        print("-" * 72)

        columns = [
            "tp",
            "sl",
            "rr",
            "horizon",
            "n",
            "win_rate",
            "net_r",
            "expectancy_r",
            "profit_factor",
            "max_dd_r",
            "positive_windows",
            "positive_window_rate",
            "min_window_net_r",
            "min_window_expectancy_r",
            "neighbor_positive_rate",
            "mean_neighbor_expectancy_r",
            "plateau_score",
        ]

        print(strategy_results[columns].head(15).to_string(index=False))

        print()

        print(f"BEST GLOBAL EXPECTANCY — {strategy_name}")

        best_expectancy = strategy_results.sort_values(
            "expectancy_r",
            ascending=False,
        ).head(5)

        print(best_expectancy[columns].to_string(index=False))

        print()

        print(f"BEST OOS CONSISTENCY — {strategy_name}")

        best_consistency = strategy_results.sort_values(
            [
                "positive_window_rate",
                "min_window_net_r",
                "expectancy_r",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).head(5)

        print(best_consistency[columns].to_string(index=False))

        print()

    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(output_summary)
    print(output_windows)
    print(output_robust)

    print()

    print("=" * 72)
    print("08Z PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
