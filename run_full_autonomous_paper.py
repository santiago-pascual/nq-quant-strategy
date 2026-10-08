from __future__ import annotations

import sys

from pathlib import Path

from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed

import os

import time

import numpy as np

import pandas as pd


# =============================================================================

# PROJECT ROOT

# =============================================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================

# PROJECT IMPORTS

# =============================================================================

from src.data_loader import load_data

from src.models.regime import VolatilityRegimeModel

from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)

from src.research.direction.direction_features import (
    add_directional_features,
)

from src.strategies.base import StrategySignal

from src.strategies.mean_reversion.backtest import (
    MeanReversionBacktestAdapter,
)

from src.strategies.mean_reversion.config import (
    MRL1_CONFIG,
    MRS2_CONFIG,
)

from src.strategies.s2r import (
    S2RStrategy,
    fit_s2_model,
)

from src.strategies.s2r.recovery import (
    RecoveryState,
)

from src.strategies.orb.config import ORBConfig

from src.strategies.orb.runner import (
    ORBBacktestRunner,
)


# =============================================================================

# CONFIG

# =============================================================================

START_DATE = pd.Timestamp(
    "2020-06-23",
    tz="UTC",
)

N_MR_WINDOWS = 22

HMM_FEATURES = [
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
]

N_HMM_STATES = 3

HMM_RANDOM_STATE = 42

HMM_N_ITER = 200

HMM_MIN_TRAIN_VALID = 500


# S2 frozen configuration.

S2_STOP_POINTS = 25.0

S2_RR = 1.75

S2_HORIZON = 20


# =============================================================================

# HELPERS

# =============================================================================


def section(title: str) -> None:

    print()

    print("=" * 100)

    print(title)

    print("=" * 100)


def normalize_timestamp(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    if "timestamp ET" in result.columns:
        source = "timestamp ET"

    elif "timestamp" in result.columns:
        source = "timestamp"

    else:
        raise KeyError("Dataset has neither 'timestamp ET' nor 'timestamp'.")

    result["canonical_timestamp"] = pd.to_datetime(
        result[source],
        errors="raise",
        utc=True,
    )

    result = result.sort_values(
        "canonical_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    return result


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name} is missing required columns:\n" + "\n".join(missing)
        )


# =============================================================================

# LOAD RAW DATA

# =============================================================================


def load_raw() -> pd.DataFrame:

    section("1. LOADING RAW MNQ")

    data = load_data()

    if data is None or data.empty:
        raise RuntimeError("load_data() returned no data.")

    data = normalize_timestamp(data)

    data = data.loc[data["canonical_timestamp"] >= START_DATE].copy()

    data.reset_index(drop=True, inplace=True)

    print(f"Rows:       {len(data):,}")

    print(
        "Start:      ",
        data["canonical_timestamp"].min(),
    )

    print(
        "End:        ",
        data["canonical_timestamp"].max(),
    )

    require_columns(
        data,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "market_period",
        ],
        "Raw dataset",
    )

    return data


# =============================================================================

# RTH DATA

# =============================================================================


def build_rth(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    section("2. BUILDING CANONICAL RTH DATA")

    if "timestamp ET" not in raw.columns:
        raise RuntimeError("Canonical RTH construction requires 'timestamp ET'.")

    rth = raw.loc[raw["market_period"] == "RTH"].copy()

    rth["canonical_timestamp"] = pd.to_datetime(
        rth["timestamp ET"],
        errors="raise",
        utc=True,
    )

    rth = rth.sort_values(
        "canonical_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"RTH rows: {len(rth):,}")

    if rth.empty:
        raise RuntimeError("No RTH rows found.")

    return rth


# =============================================================================

# 22 OOS WINDOWS

# =============================================================================


def build_windows(
    rth: pd.DataFrame,
) -> list[np.ndarray]:

    section("3. BUILDING 22 CAUSAL WINDOWS")

    indices = np.arange(
        len(rth),
        dtype=np.int64,
    )

    blocks = np.array_split(
        indices,
        N_MR_WINDOWS,
    )

    windows = [block for block in blocks if len(block) > 0]

    print(f"Windows: {len(windows)}")

    for i, block in enumerate(
        windows,
        start=1,
    ):
        start = rth.iloc[block[0]]["canonical_timestamp"]

        end = rth.iloc[block[-1]]["canonical_timestamp"]

        print(f"Window {i:02d}: {start} -> {end} | {len(block):,} bars")

    return windows


# =============================================================================

# MEAN REVERSION FEATURES

# =============================================================================


def build_mr_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    section("4. BUILDING CANONICAL MR FEATURES")

    # This is the exact repository feature pipeline.

    features = build_mean_reversion_features(raw.copy())

    features["canonical_timestamp"] = pd.to_datetime(
        features["timestamp ET"]
        if "timestamp ET" in features.columns
        else features["timestamp"],
        errors="raise",
        utc=True,
    )

    features = features.sort_values(
        "canonical_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    require_columns(
        features,
        HMM_FEATURES,
        "MR feature dataframe",
    )

    require_columns(
        features,
        [
            "zscore_30",
            "realized_vol_30",
        ],
        "MR feature dataframe",
    )

    print(f"Feature rows: {len(features):,}")

    return features


# =============================================================================

# CAUSAL HMM

# =============================================================================


def valid_hmm(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df[HMM_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


def fit_window_hmm(
    features: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    """

    Exact repository HMM methodology for one OOS window.

    IMPORTANT:

    - training set = every feature row strictly before oos_start

    - OOS set = rows between oos_start and oos_end

    - canonical VolatilityRegimeModel is used unchanged

    """

    train = features.loc[features["canonical_timestamp"] < oos_start]

    oos = features.loc[
        (features["canonical_timestamp"] >= oos_start)
        & (features["canonical_timestamp"] <= oos_end)
    ]

    train_valid = valid_hmm(train)

    oos_valid = valid_hmm(oos)

    if len(train_valid) < HMM_MIN_TRAIN_VALID or oos_valid.empty:
        return pd.DataFrame(columns=["timestamp", "hmm_state"])

    model = VolatilityRegimeModel(
        n_states=N_HMM_STATES,
        random_state=HMM_RANDOM_STATE,
        n_iter=HMM_N_ITER,
    )

    # This is intentionally the repository model API.

    model.fit(train)

    states = model.predict_states(oos)

    return pd.DataFrame(
        {
            "timestamp": oos.loc[
                states.index,
                "canonical_timestamp",
            ].to_numpy(),
            "hmm_state": states.to_numpy(dtype=np.int8),
        }
    )


def _fit_hmm_window_task(
    number: int,
    block: np.ndarray,
    features: pd.DataFrame,
    rth: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    """Worker used to parallelize the 22 independent HMM fits."""

    oos_start = rth.iloc[block[0]]["canonical_timestamp"]

    oos_end = rth.iloc[block[-1]]["canonical_timestamp"]

    states = fit_window_hmm(
        features,
        oos_start,
        oos_end,
    )

    return number, states


def build_causal_hmm_states(
    features: pd.DataFrame,
    rth: pd.DataFrame,
    windows: list[np.ndarray],
) -> pd.DataFrame:
    """

    Build the exact 22-window benchmark HMM states.

    The expensive fits are independent, so they are executed concurrently.

    This changes wall-clock scheduling, not the statistical methodology.

    HMM_WORKERS can be overridden from PowerShell:

        $env:HMM_WORKERS="2"

        python .\run_full_autonomous_paper.py

    Default is 2 to avoid multiplying BLAS memory usage excessively.

    """

    section("5. BUILDING CAUSAL HMM STATES")

    max_workers = int(
        os.environ.get(
            "HMM_WORKERS",
            max(1, min(2, os.cpu_count() or 1)),
        )
    )

    max_workers = max(1, min(max_workers, len(windows)))

    print(f"HMM workers: {max_workers}")

    print(f"HMM iterations per fit: {HMM_N_ITER}")

    print("Methodology: canonical VolatilityRegimeModel, 22 independent fits.")

    results: dict[int, pd.DataFrame] = {}

    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fit_hmm_window_task,
                number,
                block,
                features,
                rth,
            ): number
            for number, block in enumerate(windows, start=1)
        }

        for future in as_completed(futures):
            number, states = future.result()

            results[number] = states

            elapsed = time.perf_counter() - started

            print(
                f"Window {number:02d}: "
                f"{len(states):,} HMM states "
                f"| elapsed {elapsed / 60.0:.1f} min"
            )

    parts = []

    for number in sorted(results):
        states = results[number]

        if not states.empty:
            states = states.copy()

            states["window"] = number

            parts.append(states)

    if not parts:
        raise RuntimeError("No causal HMM states generated.")

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = result.sort_values(
        ["window", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    if result[["window", "timestamp"]].duplicated().any():
        raise RuntimeError("Duplicate HMM (window,timestamp) pairs.")

    print(f"Total HMM states: {len(result):,}")

    print(f"HMM stage elapsed: {(time.perf_counter() - started) / 60.0:.1f} min")

    return result


# =============================================================================

# MERGE HMM INTO FEATURE DATA

# =============================================================================


def attach_hmm_states(
    features: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:

    result = features.copy()

    result = result.merge(
        states[
            [
                "timestamp",
                "window",
                "hmm_state",
            ]
        ],
        left_on="canonical_timestamp",
        right_on="timestamp",
        how="left",
    )

    result.drop(
        columns=["timestamp"],
        inplace=True,
    )

    return result


# =============================================================================

# VOLATILITY PERCENTILE FOR MR

# =============================================================================


def add_mr_volatility_percentile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """

    Exact causal expanding percentile, implemented with a Fenwick tree.

    The old implementation sorted the entire history again for EVERY row,

    which is O(N^2 log N) and can take hours on ~2M rows.

    Here we coordinate-compress the observed volatility values once and keep

    counts of PRIOR observations in a Fenwick tree. Therefore the percentile

    remains:

        number of prior observations <= current value

        ------------------------------------------------

                    number of prior observations

    The current observation is inserted only AFTER its percentile is computed.

    This preserves the causal definition exactly.

    """

    result = df.copy()

    valid = pd.to_numeric(
        result["realized_vol_30"],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    values = valid.to_numpy(dtype=np.float64, copy=False)

    finite = np.isfinite(values)

    if not finite.any():
        result["vol_percentile"] = np.nan

        return result

    # Exact coordinate compression of all observed values.

    unique_values = np.unique(values[finite])

    n_unique = len(unique_values)

    # Fenwick tree: tree[k] = cumulative count structure.

    tree = np.zeros(n_unique + 1, dtype=np.int64)

    def add(index: int) -> None:

        # Fenwick uses 1-based indices.

        i = index + 1

        while i <= n_unique:
            tree[i] += 1

            i += i & -i

    def prefix_count(index: int) -> int:

        # Number of values with compressed index <= index.

        i = index + 1

        total = 0

        while i > 0:
            total += tree[i]

            i -= i & -i

        return total

    percentile = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    prior_count = 0

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        compressed = int(
            np.searchsorted(
                unique_values,
                value,
                side="left",
            )
        )

        if prior_count:
            # bisect_right(history, value) == count(prior values <= value)

            percentile[i] = prefix_count(compressed) / prior_count

        add(compressed)

        prior_count += 1

    result["vol_percentile"] = percentile

    return result


def build_mr_market_rows(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    rows = []

    for row in df.itertuples(index=False):
        if pd.isna(row.hmm_state):
            continue

        if pd.isna(row.vol_percentile):
            continue

        if pd.isna(row.zscore_30):
            continue

        rows.append(
            {
                "timestamp": row.canonical_timestamp,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
                "hmm_state": int(row.hmm_state),
                "vol_percentile": float(row.vol_percentile),
                "zscore": float(row.zscore_30),
            }
        )

    return rows


# =============================================================================

# RUN MR STRATEGY

# =============================================================================


def run_mr_strategy(
    rows: list[dict[str, Any]],
    config,
) -> pd.DataFrame:

    adapter = MeanReversionBacktestAdapter(config=config)

    trades = []

    for row in rows:
        trade = adapter.process_bar(row)

        if trade is not None:
            trades.append(
                {
                    "strategy": trade.strategy_name,
                    "side": trade.side,
                    "entry_timestamp": trade.entry_timestamp,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "stop_points": trade.stop_points,
                    "target_points": trade.target_points,
                    "rr": trade.rr,
                    "horizon_bars": trade.horizon_bars,
                    "exit_reason": trade.exit_reason,
                    "bars_elapsed": trade.bars_elapsed,
                    "r_multiple": trade.r_multiple,
                }
            )

    return pd.DataFrame(trades)


# =============================================================================

# S2 FEATURE ENGINE

# =============================================================================


def build_s2_features(
    features: pd.DataFrame,
) -> pd.DataFrame:

    section("6. BUILDING S2 DIRECTIONAL FEATURES")

    result = add_directional_features(features.copy())

    require_columns(
        result,
        [
            "past_return_30",
            "directional_pressure_30",
            "close_location_30",
            "normalized_momentum_30",
            "realized_vol_30",
            "hmm_state",
        ],
        "S2 dataframe",
    )

    return result


# =============================================================================

# S2 MODEL

# =============================================================================


def fit_s2(
    df: pd.DataFrame,
):

    section("7. FITTING FROZEN S2 MODEL")

    clean = df.dropna(
        subset=[
            "hmm_state",
            "past_return_30",
            "directional_pressure_30",
            "close_location_30",
            "normalized_momentum_30",
            "realized_vol_30",
        ]
    ).copy()

    split = int(len(clean) * 0.70)

    train = clean.iloc[:split].copy()

    rows = train[
        [
            "hmm_state",
            "past_return_30",
            "directional_pressure_30",
            "close_location_30",
            "normalized_momentum_30",
            "realized_vol_30",
        ]
    ].to_dict("records")

    model = fit_s2_model(rows)

    print("S2 TRAIN rows:", len(train))

    print("S2 thresholds:")

    for key, value in model.signal_model.thresholds.items():
        print(f"  {key}: {value}")

    return model, split, clean


# =============================================================================

# S2 TRADE RESOLUTION

# =============================================================================


def resolve_s2_trade(
    df: pd.DataFrame,
    entry_index: int,
    strategy: S2RStrategy,
):

    entry_price = float(df.iloc[entry_index]["close"])

    strategy.start_trade(
        entry_price=entry_price,
        entry_bar=entry_index,
    )

    target_price = entry_price - S2_STOP_POINTS * S2_RR

    stop_price = entry_price + S2_STOP_POINTS

    last_index = min(
        entry_index + S2_HORIZON,
        len(df) - 1,
    )

    recovery_started = False

    for i in range(
        entry_index + 1,
        last_index + 1,
    ):
        row = df.iloc[i]

        high = float(row["high"])

        low = float(row["low"])

        close = float(row["close"])

        result = strategy.update_trade_from_market(
            bar_index=i,
            high=high,
            close=close,
        )

        if result.state is RecoveryState.RECOVERED:
            exit_price = entry_price - S2_STOP_POINTS * 0.20

            strategy.finish_trade()

            return {
                "entry_index": entry_index,
                "exit_index": i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "r_multiple": 0.20,
                "exit_reason": "recovery",
            }

        if result.state is RecoveryState.FAILED_TO_RECOVER:
            exit_bar = result.exit_bar

            exit_price = float(df.iloc[exit_bar]["close"])

            strategy.finish_trade()

            return {
                "entry_index": entry_index,
                "exit_index": exit_bar,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "r_multiple": (entry_price - exit_price) / S2_STOP_POINTS,
                "exit_reason": "recovery_failed",
            }

        if result.state is RecoveryState.ADVERSE:
            recovery_started = True

        if not recovery_started:
            target_hit = low <= target_price

            stop_hit = high >= stop_price

            if target_hit and stop_hit:
                strategy.finish_trade()

                return {
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "r_multiple": -1.0,
                    "exit_reason": "both_hit_stop",
                }

            if target_hit:
                strategy.finish_trade()

                return {
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": target_price,
                    "r_multiple": S2_RR,
                    "exit_reason": "target",
                }

            if stop_hit:
                strategy.finish_trade()

                return {
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "r_multiple": -1.0,
                    "exit_reason": "stop",
                }

    exit_price = float(df.iloc[last_index]["close"])

    strategy.finish_trade()

    return {
        "entry_index": entry_index,
        "exit_index": last_index,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "r_multiple": (entry_price - exit_price) / S2_STOP_POINTS,
        "exit_reason": "timeout",
    }


# =============================================================================

# RUN S2

# =============================================================================


def run_s2(
    df: pd.DataFrame,
    model,
    split: int,
    clean: pd.DataFrame,
) -> pd.DataFrame:

    section("8. RUNNING S2R")

    oos = clean.iloc[split:].reset_index(drop=True)

    strategy = S2RStrategy(fitted_model=model)

    trades = []

    i = 0

    while i < len(oos) - 1:
        row = oos.iloc[i]

        context = {
            "timestamp": row["canonical_timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "hmm_state": int(row["hmm_state"]),
            "past_return_30": float(row["past_return_30"]),
            "directional_pressure_30": float(row["directional_pressure_30"]),
            "close_location_30": float(row["close_location_30"]),
            "normalized_momentum_30": float(row["normalized_momentum_30"]),
            "realized_vol_30": float(row["realized_vol_30"]),
        }

        decision = strategy.evaluate(context)

        if decision.action.value == "enter":
            trade = resolve_s2_trade(
                oos,
                i,
                strategy,
            )

            trade["strategy"] = "S2R"

            trade["entry_timestamp"] = oos.iloc[trade["entry_index"]][
                "canonical_timestamp"
            ]

            trade["exit_timestamp"] = oos.iloc[trade["exit_index"]][
                "canonical_timestamp"
            ]

            trades.append(trade)

            i = trade["exit_index"] + 1

        else:
            i += 1

    return pd.DataFrame(trades)


# =============================================================================

# ORB

# =============================================================================


def build_orb_input(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    result = raw[
        [
            "timestamp ET",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    result.rename(
        columns={"timestamp ET": "timestamp"},
        inplace=True,
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="raise",
    )

    return result.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)


def run_orb(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    section("9. RUNNING ORB")

    orb_data = build_orb_input(raw)

    runner = ORBBacktestRunner(config=ORBConfig())

    result = runner.run(orb_data)

    print(f"ORB trades: {len(result):,}")

    return result


# =============================================================================

# COMBINE

# =============================================================================


def combine_trades(
    mr_l1: pd.DataFrame,
    mr_s2: pd.DataFrame,
    s2: pd.DataFrame,
    orb: pd.DataFrame,
) -> pd.DataFrame:

    section("10. COMBINING STRATEGY STREAMS")

    frames = []

    if not mr_l1.empty:
        x = mr_l1.copy()

        x["source"] = "MRL1"

        frames.append(x)

    if not mr_s2.empty:
        x = mr_s2.copy()

        x["source"] = "MRS2"

        frames.append(x)

    if not s2.empty:
        x = s2.copy()

        x["source"] = "S2R"

        frames.append(x)

    if not orb.empty:
        x = orb.copy()

        x["source"] = "ORB"

        frames.append(x)

    if not frames:
        raise RuntimeError("No strategy generated trades.")

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    result["entry_timestamp"] = pd.to_datetime(
        result["entry_timestamp"],
        errors="coerce",
        utc=True,
    )

    result = result.sort_values(
        [
            "entry_timestamp",
            "source",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return result


# =============================================================================

# REPORT

# =============================================================================


def summarize(
    trades: pd.DataFrame,
) -> None:

    section("11. AUTONOMOUS RAW-DATA RESULT")

    print(f"Total trades: {len(trades):,}")

    print()

    if "r_multiple" in trades.columns:
        r = pd.to_numeric(
            trades["r_multiple"],
            errors="coerce",
        ).dropna()

        print(f"Total R:      {r.sum():.6f}")

        print(f"Mean R:       {r.mean():.6f}")

        print(f"Win rate:     {(r > 0).mean():.4%}")

        losses = abs(r.loc[r < 0].sum())

        profits = r.loc[r > 0].sum()

        pf = profits / losses if losses > 0 else np.inf

        print(f"Profit factor:{pf:.6f}")

    print()

    print(trades["source"].value_counts().sort_index().to_string())


# =============================================================================

# MAIN

# =============================================================================


def main() -> None:

    section("FULL AUTONOMOUS RAW-DATA STRATEGY REPLAY")

    print("IMPORTANT:")

    print("No previous trade CSVs are loaded.")

    print("No previous HMM state CSV is loaded.")

    print("No previous result CSV is loaded.")

    print("All strategy state is rebuilt from raw MNQ.")

    # -------------------------------------------------------------------------

    # RAW

    # -------------------------------------------------------------------------

    raw = load_raw()

    # -------------------------------------------------------------------------

    # RTH

    # -------------------------------------------------------------------------

    rth = build_rth(raw)

    # -------------------------------------------------------------------------

    # WINDOWS

    # -------------------------------------------------------------------------

    windows = build_windows(rth)

    # -------------------------------------------------------------------------

    # MR FEATURES

    # -------------------------------------------------------------------------

    features = build_mr_features(raw)

    # -------------------------------------------------------------------------

    # HMM

    # -------------------------------------------------------------------------

    states = build_causal_hmm_states(
        features,
        rth,
        windows,
    )

    # -------------------------------------------------------------------------

    # MERGE

    # -------------------------------------------------------------------------

    features = attach_hmm_states(
        features,
        states,
    )

    # -------------------------------------------------------------------------

    # MR VOL PERCENTILE

    # -------------------------------------------------------------------------

    features = add_mr_volatility_percentile(features)

    # -------------------------------------------------------------------------

    # MRL1 / MRS2

    # -------------------------------------------------------------------------

    mr_rows = build_mr_market_rows(features)

    print(f"MR usable rows: {len(mr_rows):,}")

    mrl1 = run_mr_strategy(
        mr_rows,
        MRL1_CONFIG,
    )

    mrs2 = run_mr_strategy(
        mr_rows,
        MRS2_CONFIG,
    )

    print(f"MRL1 trades: {len(mrl1):,}")

    print(f"MRS2 trades: {len(mrs2):,}")

    # -------------------------------------------------------------------------

    # S2

    # -------------------------------------------------------------------------

    s2_features = build_s2_features(features)

    s2_model, s2_split, s2_clean = fit_s2(s2_features)

    s2 = run_s2(
        s2_features,
        s2_model,
        s2_split,
        s2_clean,
    )

    print(f"S2R trades: {len(s2):,}")

    # -------------------------------------------------------------------------

    # ORB

    # -------------------------------------------------------------------------

    orb = run_orb(raw)

    # -------------------------------------------------------------------------

    # COMBINE

    # -------------------------------------------------------------------------

    all_trades = combine_trades(
        mrl1,
        mrs2,
        s2,
        orb,
    )

    summarize(all_trades)

    # -------------------------------------------------------------------------

    # OUTPUT

    # -------------------------------------------------------------------------

    output = ROOT / "autonomous_raw_strategy_trades.csv"

    all_trades.to_csv(
        output,
        index=False,
    )

    print()

    print("Trades written to:")

    print(output)

    print()

    print("EXPECTED RESEARCH COUNTS")

    print("  MRL1 : ~430")

    print("  S2R  : ~520/537 depending on benchmark universe")

    print("  MRS2 : ~863")

    print("  ORB  : ~1,442")


if __name__ == "__main__":
    main()
