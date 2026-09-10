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
# SEARCH GRID
# ============================================================

TP_VALUES = [
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
]

# IMPORTANT:
# SL is allowed to be as large as TP.
# This intentionally excludes the old frozen SL=2.0 benchmark
# from the main grid because the new search is defined on the
# common 2.5 -> 50.0 grid.
SL_VALUES = [
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
]

HORIZON_VALUES = [
    2,
    3,
    5,
    8,
    12,
    20,
    30,
]


# ============================================================
# FROZEN CONTEXTS
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
# GENERAL HELPERS
# ============================================================


def ensure_utc(series: pd.Series) -> pd.Series:
    """
    Convert a timestamp Series to timezone-aware UTC.
    """
    return pd.to_datetime(series, utc=True)


def classify_vol_bucket(percentile: float) -> int:
    """
    Convert causal volatility percentile into:
        0 -> 0-20
        1 -> 20-40
        2 -> 40-60
        3 -> 60-80
        4 -> 80-100
    """
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
# CAUSAL EXPANDING PERCENTILE
# ============================================================


def causal_percentile(values: np.ndarray) -> np.ndarray:
    """
    Causal expanding percentile.

    For each observation i, calculate the percentile rank of
    values[i] using only observations available up to and
    including i.

    Implementation:
        - coordinate-compress finite values
        - Fenwick tree stores counts
        - update() receives a ZERO-BASED coordinate
        - query() receives a ZERO-BASED coordinate

    This is the corrected implementation.

    IMPORTANT:
    Do not add an extra +1 before calling update/query.
    The Fenwick tree itself performs the required +1 shift.
    """

    values = np.asarray(values, dtype=float)

    result = np.full(values.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(values)

    if not finite_mask.any():
        return result

    finite_values = values[finite_mask]

    # Coordinate compression.
    unique_values = np.unique(finite_values)

    # Fenwick tree is 1-indexed internally.
    size = len(unique_values)
    tree = np.zeros(size + 1, dtype=np.int64)

    # Mapping from actual value -> zero-based coordinate.
    coordinate = {value: idx for idx, value in enumerate(unique_values)}

    def update(index: int) -> None:
        """
        Add one observation at zero-based coordinate `index`.
        """
        i = index + 1

        while i <= size:
            tree[i] += 1
            i += i & -i

    def query(index: int) -> int:
        """
        Return count <= zero-based coordinate `index`.
        """
        if index < 0:
            return 0

        i = min(index + 1, size)

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

        # Add current observation to the causal sample.
        update(idx)
        seen += 1

        count_leq = query(idx)

        result[i] = 100.0 * count_leq / seen

    return result


# ============================================================
# RESEARCH 07 RTH RECONSTRUCTION
# ============================================================


def prepare_rth(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Research 07's RTH preparation exactly.
    """

    required = {
        "timestamp ET",
        "market_period",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns for RTH preparation: {sorted(missing)}"
        )

    rth = data.loc[data["market_period"].astype(str).eq("RTH")].copy()

    rth = rth.sort_values("timestamp ET").reset_index(drop=True)

    return rth


# ============================================================
# EXACT EVENT -> RTH INDEX VALIDATION
# ============================================================


def validate_event_mapping(
    event_metadata: pd.DataFrame,
    rth: pd.DataFrame,
) -> None:
    """
    Validate the exact Research 07 event -> RTH index mapping.

    Research 07:
        event_id   = row index in event metadata
        data_index = index into the prepared RTH dataframe
    """

    event_ids = event_metadata["event_id"].to_numpy(dtype=np.int64)
    data_indices = event_metadata["data_index"].to_numpy(dtype=np.int64)

    expected_event_ids = np.arange(len(event_metadata), dtype=np.int64)

    if not np.array_equal(event_ids, expected_event_ids):
        raise RuntimeError("Research 07 event_id is not exactly equal to row index.")

    if len(data_indices) == 0:
        raise RuntimeError("Research 07 contains no events.")

    if data_indices.min() < 0:
        raise RuntimeError("Negative data_index found.")

    if data_indices.max() >= len(rth):
        raise RuntimeError("Research 07 data_index exceeds prepared RTH dataframe.")

    print(
        f"[08Y] event_id == row index: "
        f"{np.sum(event_ids == expected_event_ids)}/{len(event_ids)}"
    )

    print(f"[08Y] data_index range: {data_indices.min()} -> {data_indices.max()}")

    print(f"[08Y] RTH row index range: 0 -> {len(rth) - 1}")

    print("[08Y] exact event -> RTH mapping: PASS")


# ============================================================
# EVENT CONTEXT
# ============================================================


def build_event_context(
    event_metadata: pd.DataFrame,
    hmm_states: pd.DataFrame,
    rth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build event-level context:

        event_id
        data_index
        timestamp
        close
        zscore_30
        hmm_state
        realized_vol_30
        vol_percentile
        vol_bucket
    """

    events = event_metadata.copy()

    # --------------------------------------------------------
    # Normalize timestamps.
    # --------------------------------------------------------

    events["timestamp"] = ensure_utc(events["timestamp"])

    hmm = hmm_states.copy()

    if "timestamp" not in hmm.columns:
        raise ValueError("HMM cache must contain a timestamp column.")

    hmm["timestamp"] = ensure_utc(hmm["timestamp"])

    # --------------------------------------------------------
    # Locate volatility column.
    # --------------------------------------------------------

    possible_vol_columns = [
        "realized_vol_30",
        "realized_vol",
        "volatility",
    ]

    vol_column = None

    for column in possible_vol_columns:
        if column in rth.columns:
            vol_column = column
            break

    if vol_column is None:
        raise ValueError(
            "Could not find realized volatility column in RTH data. "
            f"Available columns: {list(rth.columns)}"
        )

    # --------------------------------------------------------
    # Causal volatility percentile on exact RTH sequence.
    # --------------------------------------------------------

    volatility = rth[vol_column].to_numpy(dtype=float)

    print(
        f"[08Y] Calculating causal percentile over "
        f"{len(volatility):,} RTH observations..."
    )

    vol_percentile = causal_percentile(volatility)

    # --------------------------------------------------------
    # Build RTH volatility dataframe.
    # --------------------------------------------------------

    rth_context = pd.DataFrame(
        {
            "timestamp": ensure_utc(rth["timestamp ET"]),
            "realized_vol_30": volatility,
            "vol_percentile": vol_percentile,
        }
    )

    rth_context["vol_bucket"] = (
        rth_context["vol_percentile"].map(classify_vol_bucket).astype(np.int16)
    )

    # --------------------------------------------------------
    # Event -> RTH exact positional mapping.
    #
    # This is preferable to merge_asof because Research 07
    # already gives the exact original RTH row.
    # --------------------------------------------------------

    data_indices = events["data_index"].to_numpy(dtype=np.int64)

    events["realized_vol_30"] = volatility[data_indices]

    events["vol_percentile"] = vol_percentile[data_indices]

    events["vol_bucket"] = (
        events["vol_percentile"].map(classify_vol_bucket).astype(np.int16)
    )

    # --------------------------------------------------------
    # HMM states.
    #
    # The HMM cache has one row per Research 07 event.
    # Match by event_id when available.
    # --------------------------------------------------------

    if "event_id" in hmm.columns:
        hmm_small = hmm[["event_id", "hmm_state"]].copy()

        events = events.merge(
            hmm_small,
            on="event_id",
            how="left",
            validate="one_to_one",
        )

    elif len(hmm) == len(events):
        events["hmm_state"] = hmm["hmm_state"].to_numpy(dtype=np.int16)

    else:
        raise ValueError("Could not align HMM states with Research 07 events.")

    events["hmm_state"] = pd.to_numeric(
        events["hmm_state"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Basic validation.
    # --------------------------------------------------------

    if events["hmm_state"].isna().any():
        missing = int(events["hmm_state"].isna().sum())

        raise RuntimeError(f"Missing HMM states for {missing} events.")

    if events["vol_bucket"].eq(-1).any():
        missing = int(events["vol_bucket"].eq(-1).sum())

        raise RuntimeError(f"Missing volatility buckets for {missing} events.")

    return events


# ============================================================
# EVENT SELECTION
# ============================================================


def select_context_events(
    events: pd.DataFrame,
    context: Dict,
) -> pd.DataFrame:
    """
    Select events belonging to a frozen strategy context.

    LONG:
        zscore <= -threshold

    SHORT:
        zscore >= +threshold
    """

    mask = events["hmm_state"].astype(int).eq(context["hmm_state"]) & events[
        "vol_bucket"
    ].astype(int).eq(context["vol_bucket"])

    if context["side"] == "LONG":
        mask &= events["zscore_30"] <= -context["z_threshold"]

    elif context["side"] == "SHORT":
        mask &= events["zscore_30"] >= context["z_threshold"]

    else:
        raise ValueError(f"Unknown side: {context['side']}")

    selected = events.loc[mask].copy()

    selected = selected.sort_values("event_id").reset_index(drop=True)

    return selected


# ============================================================
# FUTURE OHLC ARRAYS
# ============================================================


def build_future_arrays(
    rth: pd.DataFrame,
    event_data_indices: np.ndarray,
    max_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build future OHLC arrays starting from the BAR AFTER
    the signal bar.

    Shape:
        (n_events, max_horizon)

    Each row:
        t+1, t+2, ..., t+max_horizon
    """

    highs = rth["high"].to_numpy(dtype=float)
    lows = rth["low"].to_numpy(dtype=float)
    closes = rth["close"].to_numpy(dtype=float)

    offsets = np.arange(
        1,
        max_horizon + 1,
        dtype=np.int64,
    )

    indices = event_data_indices[:, None] + offsets[None, :]

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
# INTRABAR TRADE EVALUATION
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate all events for one configuration.

    Intrabar methodology:

    LONG:
        TP if high - entry >= TP
        SL if low - entry <= -SL

    SHORT:
        TP if entry - low >= TP
        SL if high - entry >= SL

    If TP and SL are both touched inside the same bar:
        STOP FIRST

    If neither barrier is hit:
        TIMEOUT

    Timeout R:
        final close P&L / SL

    Returns:
        result_code
            +1 = WIN
             0 = TIMEOUT
            -1 = LOSS

        r_multiple

        bars_to_result
            1..H for barrier hit
            H for timeout
    """

    n_events = len(entries)

    result = np.zeros(n_events, dtype=np.int8)

    r_multiple = np.zeros(
        n_events,
        dtype=np.float64,
    )

    bars_to_result = np.full(
        n_events,
        horizon,
        dtype=np.int16,
    )

    highs = future_high[:, :horizon]
    lows = future_low[:, :horizon]
    closes = future_close[:, :horizon]

    if side == "LONG":
        favorable = highs - entries[:, None]
        adverse = entries[:, None] - lows

        target_hit = favorable >= tp

        stop_hit = adverse >= sl

    elif side == "SHORT":
        favorable = entries[:, None] - lows
        adverse = highs - entries[:, None]

        target_hit = favorable >= tp

        stop_hit = adverse >= sl

    else:
        raise ValueError(f"Unknown side: {side}")

    valid_target = target_hit & np.isfinite(favorable)

    valid_stop = stop_hit & np.isfinite(adverse)

    any_target = valid_target.any(axis=1)
    any_stop = valid_stop.any(axis=1)

    # --------------------------------------------------------
    # First target / first stop.
    # --------------------------------------------------------

    first_target = np.full(
        n_events,
        horizon + 1,
        dtype=np.int16,
    )

    first_stop = np.full(
        n_events,
        horizon + 1,
        dtype=np.int16,
    )

    target_rows = np.where(any_target)[0]

    if len(target_rows) > 0:
        first_target[target_rows] = (
            np.argmax(
                valid_target[target_rows],
                axis=1,
            )
            + 1
        )

    stop_rows = np.where(any_stop)[0]

    if len(stop_rows) > 0:
        first_stop[stop_rows] = (
            np.argmax(
                valid_stop[stop_rows],
                axis=1,
            )
            + 1
        )

    # --------------------------------------------------------
    # Barrier resolution.
    #
    # STOP FIRST when both are touched on the same bar.
    # --------------------------------------------------------

    win_mask = any_target & (first_target < first_stop)

    loss_mask = any_stop & (first_stop <= first_target)

    result[win_mask] = 1
    result[loss_mask] = -1

    r_multiple[win_mask] = tp / sl
    r_multiple[loss_mask] = -1.0

    barrier_mask = win_mask | loss_mask

    barrier_rows = np.where(barrier_mask)[0]

    if len(barrier_rows) > 0:
        bars_to_result[barrier_rows] = np.where(
            win_mask[barrier_rows],
            first_target[barrier_rows],
            first_stop[barrier_rows],
        )

    # --------------------------------------------------------
    # TIMEOUT.
    #
    # Use final close P&L divided by SL.
    # --------------------------------------------------------

    timeout_mask = ~barrier_mask

    timeout_rows = np.where(timeout_mask)[0]

    if len(timeout_rows) > 0:
        final_close = closes[
            timeout_rows,
            horizon - 1,
        ]

        finite_timeout = np.isfinite(final_close)

        valid_rows = timeout_rows[finite_timeout]

        if side == "LONG":
            pnl_points = final_close[finite_timeout] - entries[valid_rows]

        else:
            pnl_points = entries[valid_rows] - final_close[finite_timeout]

        r_multiple[valid_rows] = pnl_points / sl

    return (
        result,
        r_multiple,
        bars_to_result,
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================


def max_drawdown(values: np.ndarray) -> float:
    """
    Maximum drawdown of cumulative R.
    """

    if len(values) == 0:
        return 0.0

    cumulative = np.cumsum(values)

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


def profit_factor(r_values: np.ndarray) -> float:
    """
    Gross profit / gross loss.
    """

    gains = r_values[r_values > 0].sum()

    losses = -r_values[r_values < 0].sum()

    if losses <= 0:
        return np.inf if gains > 0 else np.nan

    return float(gains / losses)


def trade_sharpe(r_values: np.ndarray) -> float:
    """
    Simple trade-level Sharpe without annualization.
    """

    if len(r_values) < 2:
        return np.nan

    std = np.std(
        r_values,
        ddof=1,
    )

    if std == 0:
        return np.nan

    return float(np.mean(r_values) / std)


def summarize_trades(
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
            "trade_sharpe": np.nan,
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
        "trade_sharpe": trade_sharpe(r_values),
    }


# ============================================================
# WINDOW EVALUATION
# ============================================================


def evaluate_windows(
    event_ids: np.ndarray,
    windows: np.ndarray,
    r_values: np.ndarray,
) -> List[Dict]:
    """
    Aggregate one configuration by Research 07 OOS window.
    """

    rows: List[Dict] = []

    unique_windows = np.sort(np.unique(windows))

    for window in unique_windows:
        mask = windows == window

        if not np.any(mask):
            continue

        summary = summarize_trades(r_values[mask])

        rows.append(
            {
                "window": int(window),
                **summary,
            }
        )

    return rows


# ============================================================
# RR BAND
# ============================================================


def rr_band(rr: float) -> str:

    if rr < 1.0:
        return "<1R"

    if rr < 1.5:
        return "1-1.5R"

    if rr < 2.0:
        return "1.5-2R"

    if rr < 3.0:
        return "2-3R"

    if rr < 5.0:
        return "3-5R"

    if rr < 10.0:
        return "5-10R"

    return ">=10R"


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 72)
    print("08Y — INTRABAR TP/SL SEARCH")
    print("=" * 72)

    print()
    print(f"TP values:      {len(TP_VALUES)}")
    print(f"SL values:      {len(SL_VALUES)}")
    print(f"Horizons:       {len(HORIZON_VALUES)}")

    configs_per_context = len(TP_VALUES) * len(SL_VALUES) * len(HORIZON_VALUES)

    total_configs = configs_per_context * len(CONTEXTS)

    print(f"Configs/context: {configs_per_context}")

    print(f"Total configs:   {total_configs}")

    print()

    # ========================================================
    # LOAD RESEARCH 07 EVENTS
    # ========================================================

    print("[08Y] Loading Research 07 event metadata...")

    events = pd.read_csv(RESEARCH_07_EVENTS)

    print(f"[08Y] Research 07 events: {len(events):,}")

    # ========================================================
    # LOAD HMM
    # ========================================================

    print("[08Y] Loading causal HMM states...")

    hmm_states = pd.read_csv(RESEARCH_08B_HMM)

    print(f"[08Y] HMM rows: {len(hmm_states):,}")

    # ========================================================
    # LOAD MARKET DATA
    # ========================================================

    print("[08Y] Loading market data...")

    market_data = load_data()

    print(f"[08Y] Market rows: {len(market_data):,}")

    # ========================================================
    # PREPARE RTH
    # ========================================================

    print("[08Y] Preparing exact Research 07 RTH...")

    rth = prepare_rth(market_data)

    print(f"[08Y] RTH rows: {len(rth):,}")

    # ========================================================
    # VALIDATE MAPPING
    # ========================================================

    validate_event_mapping(
        events,
        rth,
    )

    # ========================================================
    # BUILD EVENT CONTEXT
    # ========================================================

    print("[08Y] Building event volatility/HMM context...")

    events = build_event_context(
        events,
        hmm_states,
        rth,
    )

    print("[08Y] Volatility bucket distribution:")

    bucket_counts = events["vol_bucket"].value_counts().sort_index()

    for bucket, count in bucket_counts.items():
        labels = {
            0: "VOL0-20",
            1: "VOL20-40",
            2: "VOL40-60",
            3: "VOL60-80",
            4: "VOL80-100",
        }

        print(f"    {labels.get(int(bucket), str(bucket)):10s} {int(count):,}")

    print()

    # ========================================================
    # BUILD FUTURE ARRAYS
    # ========================================================

    max_horizon = max(HORIZON_VALUES)

    event_data_indices = events["data_index"].to_numpy(dtype=np.int64)

    print(f"[08Y] Building future OHLC arrays to H={max_horizon}...")

    (
        future_high_all,
        future_low_all,
        future_close_all,
    ) = build_future_arrays(
        rth,
        event_data_indices,
        max_horizon,
    )

    print(f"[08Y] future_high shape: {future_high_all.shape}")

    print(f"[08Y] future_low shape: {future_low_all.shape}")

    print(f"[08Y] future_close shape: {future_close_all.shape}")

    # ========================================================
    # RESULT CONTAINERS
    # ========================================================

    summary_rows: List[Dict] = []
    window_rows: List[Dict] = []
    trade_rows: List[Dict] = []

    # ========================================================
    # RUN CONTEXTS
    # ========================================================

    completed_configs = 0

    for strategy_name, context in CONTEXTS.items():
        print()
        print("-" * 72)
        print(f"[08Y] CONTEXT: {strategy_name}")
        print("-" * 72)

        selected = select_context_events(
            events,
            context,
        )

        print(f"[08Y] {strategy_name} events: {len(selected):,}")

        if len(selected) == 0:
            print(f"[08Y] WARNING: no events for {strategy_name}")
            continue

        event_ids = selected["event_id"].to_numpy(dtype=np.int64)

        data_indices = selected["data_index"].to_numpy(dtype=np.int64)

        windows = selected["window"].to_numpy(dtype=np.int64)

        timestamps = selected["timestamp"].astype(str).to_numpy()

        entries = selected["close"].to_numpy(dtype=float)

        # Direct positional access into exact Research 07
        # future arrays.
        future_high = future_high_all[event_ids]

        future_low = future_low_all[event_ids]

        future_close = future_close_all[event_ids]

        context_config_count = len(TP_VALUES) * len(SL_VALUES) * len(HORIZON_VALUES)

        context_completed = 0

        for tp in TP_VALUES:
            for sl in SL_VALUES:
                for horizon in HORIZON_VALUES:
                    (
                        result,
                        r_values,
                        bars_to_result,
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

                    metrics = summarize_trades(r_values)

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
                            "rr_band": rr_band(rr),
                            "horizon": horizon,
                            **metrics,
                        }
                    )

                    # ------------------------------------------------
                    # Window results
                    # ------------------------------------------------

                    per_window = evaluate_windows(
                        event_ids,
                        windows,
                        r_values,
                    )

                    for row in per_window:
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

                    # ------------------------------------------------
                    # Trade-level results
                    # ------------------------------------------------

                    result_labels = np.where(
                        result > 0,
                        "WIN",
                        np.where(
                            result < 0,
                            "LOSS",
                            "TIMEOUT",
                        ),
                    )

                    for i in range(len(selected)):
                        trade_rows.append(
                            {
                                "strategy_name": strategy_name,
                                "candidate_id": context["candidate_id"],
                                "side": context["side"],
                                "hmm_state": context["hmm_state"],
                                "vol_bucket": context["vol_bucket"],
                                "z_threshold": context["z_threshold"],
                                "event_id": int(event_ids[i]),
                                "data_index": int(data_indices[i]),
                                "window": int(windows[i]),
                                "timestamp": timestamps[i],
                                "entry": float(entries[i]),
                                "tp": float(tp),
                                "sl": float(sl),
                                "rr": float(rr),
                                "horizon": int(horizon),
                                "result": result_labels[i],
                                "r": float(r_values[i]),
                                "bars_to_result": int(bars_to_result[i]),
                            }
                        )

                    context_completed += 1
                    completed_configs += 1

                    if (
                        context_completed == 1
                        or context_completed % 250 == 0
                        or context_completed == context_config_count
                    ):
                        print(
                            f"[08Y] {strategy_name}: "
                            f"{context_completed:,}/"
                            f"{context_config_count:,} configs"
                        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)

    window_df = pd.DataFrame(window_rows)

    trades_df = pd.DataFrame(trade_rows)

    # ========================================================
    # SORTING
    # ========================================================

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            [
                "strategy_name",
                "expectancy_r",
                "profit_factor",
                "net_r",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        ).reset_index(drop=True)

    if not window_df.empty:
        window_df = window_df.sort_values(
            [
                "strategy_name",
                "tp",
                "sl",
                "horizon",
                "window",
            ]
        ).reset_index(drop=True)

    if not trades_df.empty:
        trades_df = trades_df.sort_values(
            [
                "strategy_name",
                "tp",
                "sl",
                "horizon",
                "event_id",
            ]
        ).reset_index(drop=True)

    # ========================================================
    # SYMMETRIC TP == SL SUMMARY
    # ========================================================

    symmetric_df = summary_df.loc[
        np.isclose(
            summary_df["tp"],
            summary_df["sl"],
        )
    ].copy()

    symmetric_df = symmetric_df.sort_values(
        [
            "strategy_name",
            "expectancy_r",
            "profit_factor",
            "net_r",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    # ========================================================
    # RR SUMMARY
    # ========================================================

    rr_summary = summary_df.groupby(
        [
            "strategy_name",
            "rr_band",
        ],
        as_index=False,
    ).agg(
        configs=(
            "rr",
            "count",
        ),
        mean_expectancy_r=(
            "expectancy_r",
            "mean",
        ),
        median_expectancy_r=(
            "expectancy_r",
            "median",
        ),
        max_expectancy_r=(
            "expectancy_r",
            "max",
        ),
        min_expectancy_r=(
            "expectancy_r",
            "min",
        ),
        mean_profit_factor=(
            "profit_factor",
            "mean",
        ),
        max_profit_factor=(
            "profit_factor",
            "max",
        ),
        positive_configs=(
            "expectancy_r",
            lambda x: int(np.sum(x > 0)),
        ),
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_summary = RESULTS_DIR / "research_08y_intrabar_tp_sl_search.csv"

    output_windows = RESULTS_DIR / "research_08y_window_results.csv"

    output_rr = RESULTS_DIR / "research_08y_rr_summary.csv"

    output_symmetric = RESULTS_DIR / "research_08y_symmetric_tp_sl.csv"

    output_trades = RESULTS_DIR / "research_08y_trade_results.csv"

    # ========================================================
    # SAVE
    # ========================================================

    summary_df.to_csv(
        output_summary,
        index=False,
    )

    window_df.to_csv(
        output_windows,
        index=False,
    )

    rr_summary.to_csv(
        output_rr,
        index=False,
    )

    symmetric_df.to_csv(
        output_symmetric,
        index=False,
    )

    trades_df.to_csv(
        output_trades,
        index=False,
    )

    # ========================================================
    # PRINT TOP RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("08Y COMPLETE")
    print("=" * 72)

    print(f"Completed configs: {completed_configs:,}")

    print()

    for strategy_name in CONTEXTS:
        strategy_results = summary_df.loc[
            summary_df["strategy_name"] == strategy_name
        ].copy()

        if strategy_results.empty:
            continue

        print("-" * 72)
        print(f"TOP RESULTS — {strategy_name}")
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
        ]

        print(strategy_results[columns].head(10).to_string(index=False))

        print()

        print(f"BEST SYMMETRIC TP=SL — {strategy_name}")

        symmetric_strategy = symmetric_df.loc[
            symmetric_df["strategy_name"] == strategy_name
        ]

        if symmetric_strategy.empty:
            print("    No symmetric configurations.")

        else:
            print(symmetric_strategy[columns].head(10).to_string(index=False))

        print()

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(output_summary)

    print(output_windows)

    print(output_rr)

    print(output_symmetric)

    print(output_trades)

    print()
    print("=" * 72)
    print("08Y PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
