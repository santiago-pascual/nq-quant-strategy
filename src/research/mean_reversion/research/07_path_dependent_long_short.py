"""
MEAN REVERSION — RESEARCH 07.1
==============================

PATH-DEPENDENT LONG / SHORT ANALYSIS

Objetivo
--------
Evaluar eventos de mean reversion utilizando el camino REAL posterior
de cada entrada.

Se prueban:

    LONG / SHORT
    Z-score
    Stop Loss
    Take Profit
    Horizonte temporal

Matriz:

    2 sides
    × 4 Z-scores
    × 7 stops
    × 9 targets
    × 5 horizons

    = 2,520 combinaciones

IMPORTANTE
----------
El futuro de cada evento se calcula UNA SOLA VEZ.

Después todas las combinaciones reutilizan ese path.

Esto evita recalcular millones de veces los mismos datos.

No:
    - XGBoost
    - optimización
    - selección automática de estrategia
    - optimización de ejecución

Same-bar:
----------
Si TP y SL son alcanzados en la misma vela de 1 minuto,
no podemos saber cuál ocurrió primero utilizando solamente OHLC.

Regla conservadora:

    STOP FIRST

Costos:
-------
No se aplican costos en este research.
Los costos reales son fijos y se incorporarán posteriormente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ZSCORE_THRESHOLDS = (
    1.5,
    2.0,
    2.5,
    3.0,
)

TARGET_POINTS = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
    75.0,
    100.0,
)

STOP_POINTS = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
)

HORIZONS = (
    10,
    20,
    30,
    60,
    120,
)

MAX_HORIZON = max(HORIZONS)

N_OOS_WINDOWS = 22


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"

CACHE_DIR = RESULTS_DIR / "cache"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PATH_CACHE_FILE = CACHE_DIR / "research_07_future_path_cache.npz"

EVENT_METADATA_FILE = CACHE_DIR / "research_07_event_metadata.csv"

WINDOW_SUMMARY_FILE = RESULTS_DIR / "research_07_path_dependent_window_summary.csv"

COMBINATION_SUMMARY_FILE = (
    RESULTS_DIR / "research_07_path_dependent_combination_summary.csv"
)

ZSCORE_SUMMARY_FILE = RESULTS_DIR / "research_07_path_dependent_zscore_summary.csv"


# =============================================================================
# DISPLAY
# =============================================================================


def separator(
    char: str = "=",
    length: int = 100,
) -> None:
    print(char * length)


def section(
    title: str,
) -> None:
    print()
    separator()
    print(title)
    separator()


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_rth(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only RTH data.
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
        raise KeyError("Missing required columns: " + ", ".join(sorted(missing)))

    result = data.loc[data["market_period"].eq("RTH")].copy()

    result = result.sort_values("timestamp ET")

    result = result.reset_index(drop=True)

    return result


def build_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete validated feature set.
    """

    result = build_mean_reversion_features(data.copy())

    if "zscore_30" not in result.columns:
        raise KeyError("Feature engine did not create zscore_30.")

    return result


# =============================================================================
# OOS WINDOWS
# =============================================================================


def build_oos_windows(
    data: pd.DataFrame,
    n_windows: int,
) -> list[np.ndarray]:
    """
    Split the complete chronological RTH dataset into exactly
    n OOS windows.

    No rows are discarded.
    """

    indices = np.arange(
        len(data),
        dtype=np.int64,
    )

    blocks = np.array_split(
        indices,
        n_windows,
    )

    return [block for block in blocks if len(block) > 0]


def assign_window_labels(
    data_length: int,
    windows: list[np.ndarray],
) -> np.ndarray:
    """
    Assign each row to its OOS window.
    """

    labels = np.zeros(
        data_length,
        dtype=np.int16,
    )

    for number, indices in enumerate(
        windows,
        start=1,
    ):
        labels[indices] = number

    return labels


# =============================================================================
# EVENT METADATA
# =============================================================================


def build_event_metadata(
    data: pd.DataFrame,
    windows: list[np.ndarray],
) -> pd.DataFrame:
    """
    Create the master event table.

    Every row with a valid zscore_30 is stored.

    Z-score thresholds are applied later.

    Therefore:

        Z=1.5
        Z=2.0
        Z=2.5
        Z=3.0

    all reuse exactly the same future path cache.
    """

    window_labels = assign_window_labels(
        len(data),
        windows,
    )

    valid = data["zscore_30"].notna().to_numpy(dtype=bool).copy()

    valid = valid & (window_labels > 0)

    indices = np.flatnonzero(valid)

    event_metadata = pd.DataFrame(
        {
            "event_id": np.arange(
                len(indices),
                dtype=np.int64,
            ),
            "data_index": indices,
            "window": (window_labels[indices]),
            "timestamp": (data.iloc[indices]["timestamp ET"].to_numpy()),
            "close": (data.iloc[indices]["close"].to_numpy(dtype=np.float32)),
            "zscore_30": (data.iloc[indices]["zscore_30"].to_numpy(dtype=np.float32)),
        }
    )

    return event_metadata


# =============================================================================
# FUTURE PATH CACHE
# =============================================================================


def build_future_path_cache(
    data: pd.DataFrame,
    event_metadata: pd.DataFrame,
    max_horizon: int,
) -> dict[str, np.ndarray]:
    """
    Build future paths ONCE.

    For every event we store the next max_horizon bars.

    We store distances from entry instead of repeatedly working
    with absolute prices.

    LONG:

        favorable = future high - entry
        adverse   = entry - future low

    SHORT:

        favorable = entry - future low
        adverse   = future high - entry
    """

    event_indices = event_metadata["data_index"].to_numpy(dtype=np.int64)

    entry_prices = event_metadata["close"].to_numpy(dtype=np.float32)

    n_events = len(event_indices)

    print(f"Events in cache: {n_events:,}")

    print(f"Maximum horizon: {max_horizon}")

    # -------------------------------------------------------------------------
    # Raw market arrays.
    # -------------------------------------------------------------------------

    high = data["high"].to_numpy(dtype=np.float32)

    low = data["low"].to_numpy(dtype=np.float32)

    close = data["close"].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------------------
    # Allocate future OHLC matrices.
    # -------------------------------------------------------------------------

    future_high = np.empty(
        (
            n_events,
            max_horizon,
        ),
        dtype=np.float32,
    )

    future_low = np.empty(
        (
            n_events,
            max_horizon,
        ),
        dtype=np.float32,
    )

    future_close = np.empty(
        (
            n_events,
            max_horizon,
        ),
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Future indices.
    # -------------------------------------------------------------------------

    offsets = np.arange(
        1,
        max_horizon + 1,
        dtype=np.int64,
    )

    index_matrix = event_indices[:, None] + offsets[None, :]

    valid = index_matrix < len(data)

    safe_indices = np.minimum(
        index_matrix,
        len(data) - 1,
    )

    future_high[:, :] = high[safe_indices]

    future_low[:, :] = low[safe_indices]

    future_close[:, :] = close[safe_indices]

    future_high[~valid] = np.nan

    future_low[~valid] = np.nan

    future_close[~valid] = np.nan

    # -------------------------------------------------------------------------
    # Convert to point distances.
    # -------------------------------------------------------------------------

    entry = entry_prices[:, None]

    long_favorable = np.maximum(
        future_high - entry,
        0.0,
    )

    long_adverse = np.maximum(
        entry - future_low,
        0.0,
    )

    short_favorable = np.maximum(
        entry - future_low,
        0.0,
    )

    short_adverse = np.maximum(
        future_high - entry,
        0.0,
    )

    # -------------------------------------------------------------------------
    # Raw high / low no longer needed.
    # -------------------------------------------------------------------------

    del future_high
    del future_low

    return {
        "future_close": future_close,
        "long_favorable": (
            long_favorable.astype(
                np.float32,
                copy=False,
            )
        ),
        "long_adverse": (
            long_adverse.astype(
                np.float32,
                copy=False,
            )
        ),
        "short_favorable": (
            short_favorable.astype(
                np.float32,
                copy=False,
            )
        ),
        "short_adverse": (
            short_adverse.astype(
                np.float32,
                copy=False,
            )
        ),
    }


def save_path_cache(
    cache: dict[str, np.ndarray],
    event_metadata: pd.DataFrame,
) -> None:
    """
    Save the reusable cache.
    """

    np.savez_compressed(
        PATH_CACHE_FILE,
        **cache,
    )

    event_metadata.to_csv(
        EVENT_METADATA_FILE,
        index=False,
    )


# =============================================================================
# EVENT SELECTION
# =============================================================================


def select_event_ids(
    event_metadata: pd.DataFrame,
    side: str,
    threshold: float,
) -> np.ndarray:
    """
    Select events according to side and Z-score.
    """

    z = event_metadata["zscore_30"].to_numpy(dtype=np.float32)

    if side == "LONG":
        mask = z <= -threshold

    elif side == "SHORT":
        mask = z >= threshold

    else:
        raise ValueError(f"Unknown side: {side}")

    return np.flatnonzero(mask).astype(np.int64)


# =============================================================================
# SINGLE PARAMETER COMBINATION
# =============================================================================


def evaluate_combination(
    event_metadata: pd.DataFrame,
    cache: dict[str, np.ndarray],
    event_ids: np.ndarray,
    side: str,
    threshold: float,
    stop_points: float,
    target_points: float,
    horizon: int,
    n_windows: int,
) -> list[dict]:
    """
    Evaluate one complete:

        side
        Z-score
        SL
        TP
        horizon

    combination.

    IMPORTANT:

    No trade-level DataFrame is created.

    Results are aggregated directly by OOS window.

    This is what keeps memory usage under control.
    """

    if len(event_ids) == 0:
        return []

    metadata = event_metadata.iloc[event_ids]

    window_numbers = metadata["window"].to_numpy(dtype=np.int16)

    entry_prices = metadata["close"].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------------------
    # Select cached path.
    # -------------------------------------------------------------------------

    if side == "LONG":
        favorable = cache["long_favorable"][
            event_ids,
            :horizon,
        ]

        adverse = cache["long_adverse"][
            event_ids,
            :horizon,
        ]

    else:
        favorable = cache["short_favorable"][
            event_ids,
            :horizon,
        ]

        adverse = cache["short_adverse"][
            event_ids,
            :horizon,
        ]

    # -------------------------------------------------------------------------
    # Determine whether target / stop are touched.
    # -------------------------------------------------------------------------

    target_hit = favorable >= target_points

    stop_hit = adverse >= stop_points

    has_target = target_hit.any(axis=1)

    has_stop = stop_hit.any(axis=1)

    # -------------------------------------------------------------------------
    # First target / stop bar.
    # -------------------------------------------------------------------------

    first_target = np.full(
        len(event_ids),
        horizon + 1,
        dtype=np.int16,
    )

    first_stop = np.full(
        len(event_ids),
        horizon + 1,
        dtype=np.int16,
    )

    if has_target.any():
        target_positions = np.flatnonzero(has_target)

        first_target[target_positions] = np.argmax(
            target_hit[target_positions],
            axis=1,
        ).astype(np.int16)

    if has_stop.any():
        stop_positions = np.flatnonzero(has_stop)

        first_stop[stop_positions] = np.argmax(
            stop_hit[stop_positions],
            axis=1,
        ).astype(np.int16)

    # -------------------------------------------------------------------------
    # Determine outcome.
    #
    # Target before stop -> WIN
    # Stop before target -> LOSS
    # Same bar           -> LOSS
    # Neither            -> TIMEOUT
    # -------------------------------------------------------------------------

    win_mask = has_target & (~has_stop | (first_target < first_stop))

    loss_mask = has_stop & (~has_target | (first_stop <= first_target))

    timeout_mask = ~(win_mask | loss_mask)

    # -------------------------------------------------------------------------
    # Result in points.
    # -------------------------------------------------------------------------

    result_points = np.zeros(
        len(event_ids),
        dtype=np.float32,
    )

    result_points[win_mask] = target_points

    result_points[loss_mask] = -stop_points

    # -------------------------------------------------------------------------
    # TIMEOUT:
    #
    # Exit at close of final horizon bar.
    # -------------------------------------------------------------------------

    if timeout_mask.any():
        timeout_positions = np.flatnonzero(timeout_mask)

        timeout_event_ids = event_ids[timeout_positions]

        timeout_close = cache["future_close"][
            timeout_event_ids,
            horizon - 1,
        ]

        if side == "LONG":
            timeout_points = timeout_close - entry_prices[timeout_positions]

        else:
            timeout_points = entry_prices[timeout_positions] - timeout_close

        result_points[timeout_positions] = timeout_points

    # -------------------------------------------------------------------------
    # MFE / MAE.
    # -------------------------------------------------------------------------

    mfe = np.max(
        favorable,
        axis=1,
    )

    mae = np.max(
        adverse,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Holding bars.
    # -------------------------------------------------------------------------

    exit_offset = np.full(
        len(event_ids),
        horizon - 1,
        dtype=np.int16,
    )

    exit_offset[win_mask] = first_target[win_mask]

    exit_offset[loss_mask] = first_stop[loss_mask]

    holding_bars = exit_offset + 1

    # -------------------------------------------------------------------------
    # Aggregate by OOS window.
    # -------------------------------------------------------------------------

    rows = []

    for window in range(
        1,
        n_windows + 1,
    ):
        window_mask = window_numbers == window

        if not window_mask.any():
            continue

        points = result_points[window_mask]

        wins = int(win_mask[window_mask].sum())

        losses = int(loss_mask[window_mask].sum())

        timeouts = int(timeout_mask[window_mask].sum())

        trades = int(window_mask.sum())

        gross_profit = points[points > 0].sum(dtype=np.float64)

        gross_loss = abs(points[points < 0].sum(dtype=np.float64))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss

        else:
            profit_factor = np.nan

        rows.append(
            {
                "side": side,
                "zscore_threshold": threshold,
                "stop_points": stop_points,
                "target_points": target_points,
                "horizon_bars": horizon,
                "window": window,
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "timeouts": timeouts,
                "win_rate": (wins / trades),
                "total_points": float(points.sum(dtype=np.float64)),
                "mean_result_points": float(points.mean()),
                "profit_factor": (
                    float(profit_factor) if np.isfinite(profit_factor) else np.nan
                ),
                "mean_mfe": float(mfe[window_mask].mean()),
                "median_mfe": float(np.median(mfe[window_mask])),
                "mean_mae": float(mae[window_mask].mean()),
                "median_mae": float(np.median(mae[window_mask])),
                "mean_holding_bars": float(holding_bars[window_mask].mean()),
            }
        )

    return rows


# =============================================================================
# COMPLETE PARAMETER SWEEP
# =============================================================================


def run_research(
    data: pd.DataFrame,
    event_metadata: pd.DataFrame,
    cache: dict[str, np.ndarray],
    windows: list[np.ndarray],
) -> pd.DataFrame:
    """
    Run all 2,520 parameter combinations.

    Results are aggregated immediately.

    No giant trade-level DataFrame is stored.
    """

    total_experiments = (
        2
        * len(ZSCORE_THRESHOLDS)
        * len(HORIZONS)
        * len(STOP_POINTS)
        * len(TARGET_POINTS)
    )

    experiment = 0

    all_rows = []

    for side in (
        "LONG",
        "SHORT",
    ):
        for threshold in ZSCORE_THRESHOLDS:
            event_ids = select_event_ids(
                event_metadata,
                side,
                threshold,
            )

            print()
            print(f"{side} | Z={threshold:.1f} | events={len(event_ids):,}")

            for horizon in HORIZONS:
                # -------------------------------------------------------------
                # Remove events without enough future bars.
                # -------------------------------------------------------------

                event_data_indices = event_metadata.iloc[event_ids][
                    "data_index"
                ].to_numpy(dtype=np.int64)

                valid = event_data_indices + horizon < len(data)

                valid_event_ids = event_ids[valid]

                for stop_points in STOP_POINTS:
                    for target_points in TARGET_POINTS:
                        experiment += 1

                        if (
                            experiment == 1
                            or experiment % 50 == 0
                            or experiment == total_experiments
                        ):
                            print(
                                f"  experiment "
                                f"{experiment:,}/"
                                f"{total_experiments:,}"
                                f" | H={horizon}"
                                f" | SL={stop_points:.0f}"
                                f" | TP={target_points:.0f}"
                            )

                        if len(valid_event_ids) == 0:
                            continue

                        rows = evaluate_combination(
                            event_metadata=event_metadata,
                            cache=cache,
                            event_ids=valid_event_ids,
                            side=side,
                            threshold=threshold,
                            stop_points=stop_points,
                            target_points=target_points,
                            horizon=horizon,
                            n_windows=len(windows),
                        )

                        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


# =============================================================================
# COMBINATION SUMMARY
# =============================================================================


def build_combination_summary(
    window_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate each parameter combination over all OOS windows.
    """

    if window_summary.empty:
        return pd.DataFrame()

    group_columns = [
        "side",
        "zscore_threshold",
        "stop_points",
        "target_points",
        "horizon_bars",
    ]

    summary = (
        window_summary.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            windows=(
                "window",
                "nunique",
            ),
            total_trades=(
                "trades",
                "sum",
            ),
            total_points=(
                "total_points",
                "sum",
            ),
            mean_window_points=(
                "total_points",
                "mean",
            ),
            median_window_points=(
                "total_points",
                "median",
            ),
            mean_win_rate=(
                "win_rate",
                "mean",
            ),
            median_win_rate=(
                "win_rate",
                "median",
            ),
            mean_profit_factor=(
                "profit_factor",
                "mean",
            ),
            median_profit_factor=(
                "profit_factor",
                "median",
            ),
            mean_mfe=(
                "mean_mfe",
                "mean",
            ),
            mean_mae=(
                "mean_mae",
                "mean",
            ),
            mean_holding_bars=(
                "mean_holding_bars",
                "mean",
            ),
        )
        .reset_index()
    )

    positive_window_ratio = (
        window_summary.assign(positive=(window_summary["total_points"] > 0))
        .groupby(
            group_columns,
            dropna=False,
        )["positive"]
        .mean()
        .reset_index(name="positive_window_ratio")
    )

    summary = summary.merge(
        positive_window_ratio,
        on=group_columns,
        how="left",
    )

    return summary


# =============================================================================
# Z-SCORE SUMMARY
# =============================================================================


def build_zscore_summary(
    combination_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive comparison of Z-score thresholds.
    """

    if combination_summary.empty:
        return pd.DataFrame()

    return (
        combination_summary.groupby(
            [
                "side",
                "zscore_threshold",
            ],
            dropna=False,
        )
        .agg(
            combinations=(
                "target_points",
                "count",
            ),
            mean_positive_window_ratio=(
                "positive_window_ratio",
                "mean",
            ),
            median_positive_window_ratio=(
                "positive_window_ratio",
                "median",
            ),
            mean_win_rate=(
                "mean_win_rate",
                "mean",
            ),
            mean_profit_factor=(
                "mean_profit_factor",
                "mean",
            ),
            mean_total_points=(
                "total_points",
                "mean",
            ),
            mean_mfe=(
                "mean_mfe",
                "mean",
            ),
            mean_mae=(
                "mean_mae",
                "mean",
            ),
        )
        .reset_index()
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    section("MEAN REVERSION — RESEARCH 07.1")

    print("MEMORY-EFFICIENT PATH-DEPENDENT LONG / SHORT ANALYSIS")

    print()
    print("No XGBoost.")

    print("No parameter optimization.")

    print("No final strategy.")

    print()
    print("Same-bar TARGET + STOP -> STOP")

    print()
    print(f"Z-score thresholds: {ZSCORE_THRESHOLDS}")

    print(f"Targets: {TARGET_POINTS}")

    print(f"Stops: {STOP_POINTS}")

    print(f"Horizons: {HORIZONS}")

    print(f"OOS windows: {N_OOS_WINDOWS}")

    # =========================================================================
    # LOAD
    # =========================================================================

    section("LOADING MNQ DATA")

    data = load_data()

    print(f"Rows loaded: {len(data):,}")

    # =========================================================================
    # RTH
    # =========================================================================

    section("PREPARING RTH")

    data = prepare_rth(data)

    print(f"RTH rows: {len(data):,}")

    # =========================================================================
    # FEATURES
    # =========================================================================

    section("BUILDING FEATURES")

    data = build_features(data)

    print(f"Feature columns: {len(data.columns):,}")

    # =========================================================================
    # OOS WINDOWS
    # =========================================================================

    section("BUILDING OOS WINDOWS")

    windows = build_oos_windows(
        data,
        N_OOS_WINDOWS,
    )

    print(f"OOS windows: {len(windows)}")

    # =========================================================================
    # EVENTS
    # =========================================================================

    section("BUILDING MASTER EVENT TABLE")

    event_metadata = build_event_metadata(
        data,
        windows,
    )

    print(f"Master events: {len(event_metadata):,}")

    # =========================================================================
    # CACHE
    # =========================================================================

    section("BUILDING FUTURE PATH CACHE")

    print("This expensive calculation happens ONCE.")

    print("The resulting cache is reusable.")

    cache = build_future_path_cache(
        data=data,
        event_metadata=event_metadata,
        max_horizon=MAX_HORIZON,
    )

    save_path_cache(
        cache,
        event_metadata,
    )

    print()
    print("Future-path cache saved:")

    print(PATH_CACHE_FILE)

    print(EVENT_METADATA_FILE)

    # =========================================================================
    # RESEARCH
    # =========================================================================

    section("RUNNING PATH-DEPENDENT RESEARCH")

    window_summary = run_research(
        data=data,
        event_metadata=event_metadata,
        cache=cache,
        windows=windows,
    )

    if window_summary.empty:
        raise RuntimeError("Research produced no results.")

    # =========================================================================
    # SUMMARIES
    # =========================================================================

    section("BUILDING SUMMARIES")

    combination_summary = build_combination_summary(window_summary)

    zscore_summary = build_zscore_summary(combination_summary)

    # =========================================================================
    # SAVE
    # =========================================================================

    window_summary.to_csv(
        WINDOW_SUMMARY_FILE,
        index=False,
    )

    combination_summary.to_csv(
        COMBINATION_SUMMARY_FILE,
        index=False,
    )

    zscore_summary.to_csv(
        ZSCORE_SUMMARY_FILE,
        index=False,
    )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    section("RESEARCH 07.1 COMPLETE")

    print(f"Window-level rows: {len(window_summary):,}")

    print(f"Parameter combinations: {len(combination_summary):,}")

    print()
    print("FILES SAVED")

    print(WINDOW_SUMMARY_FILE)

    print(COMBINATION_SUMMARY_FILE)

    print(ZSCORE_SUMMARY_FILE)

    print()
    print("REUSABLE CACHE")

    print(PATH_CACHE_FILE)

    print(EVENT_METADATA_FILE)

    # =========================================================================
    # TOP COMBINATIONS
    # =========================================================================

    print()
    print("TOP DESCRIPTIVE COMBINATIONS")

    top = combination_summary.sort_values(
        [
            "positive_window_ratio",
            "total_points",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(30)

    print(
        top.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =========================================================================
    # Z-SCORE SUMMARY
    # =========================================================================

    print()
    print("Z-SCORE COMPARISON")

    print(
        zscore_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =========================================================================
    # METHODOLOGY
    # =========================================================================

    print()
    print("IMPORTANT:")

    print("Future paths are calculated once.")

    print("All SL/TP/Horizon combinations reuse the same paths.")

    print("Same-bar TARGET + STOP is classified as STOP.")

    print("No transaction costs are applied.")

    print("No parameter set is selected as a final strategy.")

    print("Results are descriptive OOS research.")

    separator()


if __name__ == "__main__":
    main()
