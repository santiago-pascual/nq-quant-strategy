from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

S2_TRADES_PATH = RESULTS_DIR / "s2_benchmark_trades.csv"
S3_PATHS_PATH = RESULTS_DIR / "s3_failure_path_enriched.csv"
S4_COHORT_PATH = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"
S26_RESULTS_PATH = RESULTS_DIR / "s26_mae_recovery_integration_trades.csv"
S27_TRADES_PATH = RESULTS_DIR / "s27_full_strategy_trades.csv"

HMM_FEATURES = (
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
)
BASE_FEATURES = (
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
)

TARGET_STATE = 2
TAIL_PERCENT = 17.5
QUALITY_THRESHOLD = 0.75
VOLATILITY_LOW = 0.40
VOLATILITY_HIGH = 0.60

STOP_POINTS = 25.0
REWARD_RISK = 1.75
HORIZON_BARS = 20
SLIPPAGE_POINTS_PER_SIDE = 0.25
ROUND_TRIP_FEE_USD = 1.22
MNQ_POINT_VALUE = 2.00
TOTAL_COST_POINTS = (
    2 * SLIPPAGE_POINTS_PER_SIDE + ROUND_TRIP_FEE_USD / MNQ_POINT_VALUE
)

MAE_THRESHOLD_R = 0.70
RECOVERY_LEVEL_R = 0.20
RECOVERY_DEADLINE_BARS = 6
S4_DECISION_BAR = 8
S4_ADVERSE_THRESHOLD_R = 0.75

BENCHMARK_START = pd.Timestamp("2020-06-23", tz="UTC")
BENCHMARK_END = pd.Timestamp("2026-06-19 23:59:59.999999", tz="UTC")


@dataclass
class ReconstructionStages:
    market: pd.DataFrame
    window_audit: pd.DataFrame
    s2_candidates: pd.DataFrame
    s2_trades: pd.DataFrame
    s3_paths: pd.DataFrame
    s4_cohort: pd.DataFrame
    s26_results: pd.DataFrame
    s27_trades: pd.DataFrame


def load_raw_feature_data() -> pd.DataFrame:
    """Load the full Databento history and apply the frozen canonical features."""

    return add_directional_features(load_data())


def prepare_rth(market: pd.DataFrame) -> pd.DataFrame:
    rth = market.copy()
    timestamps = pd.to_datetime(rth["timestamp ET"], errors="coerce", utc=True)
    rth["_timestamp_et"] = timestamps.dt.tz_convert("America/New_York")
    rth = rth.loc[rth["market_period"] == "RTH"].copy()
    rth = rth.sort_values("_timestamp_et")
    rth = rth.set_index("_timestamp_et")
    rth.index.name = "timestamp_et"
    if "session_date" in rth.columns:
        rth["_session_id"] = rth["session_date"].astype(str)
    else:
        rth["_session_id"] = rth.index.date.astype(str)
    return rth


def generate_windows(
    rth: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = rth.index.min()
    end = rth.index.max()
    validation_start = start + pd.DateOffset(years=2)
    windows = []
    while validation_start < end:
        validation_end = min(validation_start + pd.DateOffset(months=3), end)
        train_start = validation_start - pd.DateOffset(years=2)
        windows.append((train_start, validation_start, validation_end))
        validation_start += pd.DateOffset(months=3)
    return windows


def fit_signal_parameters(
    train: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    state_train = train.loc[train["hmm_state"] == TARGET_STATE]
    thresholds: dict[str, float] = {}
    scales: dict[str, float] = {}
    quantile = TAIL_PERCENT / 100.0

    for feature in BASE_FEATURES:
        values = (
            pd.to_numeric(state_train[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        threshold = float(values.quantile(quantile))
        extreme = float(values.quantile(0.05))
        scale = threshold - extreme
        thresholds[feature] = threshold
        scales[feature] = scale if scale > 0 else float("nan")

    volatility_reference = (
        pd.to_numeric(train["realized_vol_30"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values()
        .to_numpy(dtype=float)
    )
    return thresholds, scales, volatility_reference


def calculate_quality(
    row: pd.Series,
    thresholds: dict[str, float],
    scales: dict[str, float],
) -> float:
    scores = []
    for feature in BASE_FEATURES:
        value = row[feature]
        threshold = thresholds[feature]
        scale = scales[feature]
        if pd.isna(value) or pd.isna(threshold) or pd.isna(scale) or scale <= 0:
            return float("nan")
        scores.append(float(np.clip((threshold - value) / scale, 0.0, 1.0)))
    return float(np.mean(scores))


def transform_volatility(value: float, reference: np.ndarray) -> float:
    if not np.isfinite(value) or reference.size == 0:
        return float("nan")
    return float(np.searchsorted(reference, value, side="right") / len(reference))


def entry_qualifies(
    row: pd.Series,
    *,
    hmm_state: int,
    thresholds: dict[str, float],
    scales: dict[str, float],
    volatility_percentile: float,
) -> bool:
    if hmm_state != TARGET_STATE:
        return False
    for feature in BASE_FEATURES:
        value = row[feature]
        threshold = thresholds[feature]
        if pd.isna(value) or pd.isna(threshold) or value > threshold:
            return False
    quality = calculate_quality(row, thresholds, scales)
    return (
        np.isfinite(quality)
        and quality >= QUALITY_THRESHOLD
        and np.isfinite(volatility_percentile)
        and VOLATILITY_LOW <= volatility_percentile < VOLATILITY_HIGH
    )


def resolve_short_trade(
    session: pd.DataFrame,
    entry_position: int,
) -> dict[str, object]:
    close = session["close"].to_numpy(dtype=float)
    high = session["high"].to_numpy(dtype=float)
    low = session["low"].to_numpy(dtype=float)
    entry_price = float(close[entry_position])
    target_points = STOP_POINTS * REWARD_RISK
    target_price = entry_price - target_points
    stop_price = entry_price + STOP_POINTS
    last_position = min(entry_position + HORIZON_BARS, len(session) - 1)

    for position in range(entry_position + 1, last_position + 1):
        target_hit = low[position] <= target_price
        stop_hit = high[position] >= stop_price
        if target_hit and stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
                "exit_position": position,
            }
        if target_hit:
            return {
                "raw_points": target_points,
                "reason": "target",
                "exit_position": position,
            }
        if stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "stop",
                "exit_position": position,
            }

    return {
        "raw_points": entry_price - float(close[last_position]),
        "reason": "timeout",
        "exit_position": last_position,
    }


def walk_signal_positions(
    positions: np.ndarray,
    session_length: int,
    qualifies: Callable[[int], bool],
    resolve_exit_position: Callable[[int], int],
    horizon: int = HORIZON_BARS,
) -> list[int]:
    """Reproduce S2's one-position-at-a-time iteration over state-2 bars."""

    selected: list[int] = []
    index = 0
    while index < len(positions):
        position = int(positions[index])
        if position >= session_length - horizon:
            break
        if not qualifies(position):
            index += 1
            continue
        selected.append(position)
        exit_position = int(resolve_exit_position(position))
        index = int(np.searchsorted(positions, exit_position, side="right"))
    return selected


def generate_s2_trades(
    rth: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    *,
    window_limit: int | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades_by_window = []
    candidates_by_window = []
    audits = []
    selected_windows = windows if window_limit is None else windows[:window_limit]

    for number, (train_start, validation_start, validation_end) in enumerate(
        selected_windows, start=1
    ):
        if progress:
            print(
                f"S2 window {number}/{len(windows)}: "
                f"train {train_start}..{validation_start}; "
                f"validation {validation_start}..{validation_end}"
            )
        # Pandas datetime label slices are inclusive at both endpoints in the
        # frozen script; the shared validation_start bar is intentionally kept.
        train = rth.loc[train_start:validation_start].copy()
        validation = rth.loc[validation_start:validation_end].copy()
        if train.empty or validation.empty:
            continue

        model = VolatilityRegimeModel(n_states=3, random_state=42)
        model.fit(train)
        train["hmm_state"] = model.predict_states(train)
        validation["hmm_state"] = model.predict_states(validation)
        thresholds, scales, volatility_reference = fit_signal_parameters(train)
        audits.append(
            {
                "window": number,
                "train_start": train_start,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "train_rows": len(train),
                "train_state2_rows": int((train["hmm_state"] == TARGET_STATE).sum()),
                "valid_volatility_rows": len(volatility_reference),
                "validation_rows": len(validation),
                **{f"threshold_{name}": value for name, value in thresholds.items()},
                **{f"scale_{name}": value for name, value in scales.items()},
            }
        )

        for session_id, session in validation.groupby("_session_id", sort=False):
            session = session.sort_index()
            if len(session) <= HORIZON_BARS:
                continue
            positions = np.flatnonzero(
                (session["hmm_state"] == TARGET_STATE).to_numpy()
            )

            def qualifies(position: int) -> bool:
                row = session.iloc[position]
                percentile = transform_volatility(
                    float(row["realized_vol_30"]), volatility_reference
                )
                return entry_qualifies(
                    row,
                    hmm_state=int(row["hmm_state"]),
                    thresholds=thresholds,
                    scales=scales,
                    volatility_percentile=percentile,
                )

            candidate_positions = []
            candidate_metadata: dict[int, tuple[float, float]] = {}
            for position in positions:
                position = int(position)
                if position >= len(session) - HORIZON_BARS:
                    break
                if not qualifies(position):
                    continue
                row = session.iloc[position]
                percentile = transform_volatility(
                    float(row["realized_vol_30"]), volatility_reference
                )
                quality = calculate_quality(row, thresholds, scales)
                candidate_positions.append(position)
                candidate_metadata[position] = (quality, percentile)
                candidates_by_window.append(
                    {
                        "timestamp": session.index[position],
                        "session_id": str(session_id),
                        "window": number,
                        "hmm_state": TARGET_STATE,
                        "quality": quality,
                        "vol_percentile": percentile,
                    }
                )

            def exit_position(position: int) -> int:
                return int(
                    resolve_short_trade(session, position)["exit_position"]
                )

            for position in walk_signal_positions(
                np.asarray(candidate_positions, dtype=int),
                len(session),
                lambda _: True,
                exit_position,
            ):
                row = session.iloc[position]
                quality, percentile = candidate_metadata[position]
                result = resolve_short_trade(session, position)
                raw_points = float(result["raw_points"])
                net_points = raw_points - TOTAL_COST_POINTS
                exit_position_value = int(result["exit_position"])
                trades_by_window.append(
                    {
                        "entry_timestamp": session.index[position],
                        "exit_timestamp": session.index[exit_position_value],
                        "session_id": str(session_id),
                        "quality": quality,
                        "vol_percentile": percentile,
                        "stop_points": STOP_POINTS,
                        "rr": REWARD_RISK,
                        "horizon": HORIZON_BARS,
                        "raw_points": raw_points,
                        "net_points": net_points,
                        "net_R": net_points / STOP_POINTS,
                        "exit_reason": result["reason"],
                        "holding_bars": exit_position_value - position,
                        "window": number,
                        "validation_start": validation_start,
                        "validation_end": validation_end,
                    }
                )

    return (
        pd.DataFrame(trades_by_window),
        pd.DataFrame(audits),
        pd.DataFrame(candidates_by_window),
    )


def build_s3_paths(
    market: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    bars = market.copy()
    timestamps = pd.to_datetime(bars["timestamp ET"], errors="coerce", utc=True)
    bars["_timestamp_et"] = timestamps.dt.tz_convert("America/New_York")
    bars = bars.sort_values("_timestamp_et")
    bars = bars.drop_duplicates(subset=["_timestamp_et"], keep="last")
    bars = bars.set_index("_timestamp_et")

    records = []
    for trade_index, trade in trades.reset_index(drop=True).iterrows():
        entry_timestamp = trade["entry_timestamp"]
        session_id = str(trade["session_id"])
        if entry_timestamp not in bars.index:
            continue
        session = bars.loc[bars["session_date"].astype(str) == session_id].copy()
        if session.empty:
            continue
        session = session.sort_index()
        positions = np.flatnonzero(session.index == entry_timestamp)
        if not len(positions):
            continue
        entry_position = int(positions[0])
        future = session.iloc[
            entry_position : min(entry_position + HORIZON_BARS + 1, len(session))
        ]
        if len(future) < 2:
            continue

        entry_price = float(future["close"].iloc[0])
        highs = future["high"].to_numpy(dtype=float)
        lows = future["low"].to_numpy(dtype=float)
        closes = future["close"].to_numpy(dtype=float)
        mae_r = (highs - entry_price) / STOP_POINTS
        mfe_r = (entry_price - lows) / STOP_POINTS
        close_r = (entry_price - closes) / STOP_POINTS
        record: dict[str, object] = {
            "trade_index": trade_index,
            "entry_price": entry_price,
            "path_length": len(future),
        }
        for bar in range(1, len(future)):
            record[f"mae_{bar}R"] = float(np.max(mae_r[1 : bar + 1]))
            record[f"mfe_{bar}R"] = float(np.max(mfe_r[1 : bar + 1]))
            record[f"close_{bar}R"] = float(close_r[bar])
        record.update(
            {
                "max_MAE_R": float(np.max(mae_r[1:])),
                "max_MFE_R": float(np.max(mfe_r[1:])),
                "final_close_R": float(close_r[-1]),
                "time_to_max_MFE": int(np.argmax(mfe_r[1:]) + 1),
                "time_to_max_MAE": int(np.argmax(mae_r[1:]) + 1),
                "outcome": "WIN" if float(trade["net_R"]) > 0 else "LOSS",
                "exit_reason": trade["exit_reason"],
                "net_R": float(trade["net_R"]),
                "window": trade["window"],
            }
        )
        records.append(record)

    paths = pd.DataFrame(records)
    if paths.empty and not trades.empty:
        raise RuntimeError("No raw intratrade paths could be recovered.")
    enriched = trades.reset_index(drop=True).copy()
    enriched["trade_index"] = enriched.index
    merged = enriched.merge(
        paths, on="trade_index", how="inner", suffixes=("", "_path")
    )
    if len(merged) != len(trades):
        raise RuntimeError(
            f"S3 recovered {len(merged)} of {len(trades)} raw trade paths."
        )
    return merged


def build_s4_cohort(s3_paths: pd.DataFrame) -> pd.DataFrame:
    df = s3_paths.copy()
    for bar in range(1, S4_DECISION_BAR + 1):
        for metric in ("mae", "mfe", "close"):
            column = f"{metric}_{bar}R"
            df[f"{metric}_{bar}"] = pd.to_numeric(df[column], errors="coerce")

    for metric in ("mae", "mfe", "close"):
        for start_bar in (1, 2, 3, 5):
            df[f"{metric}_change_{start_bar}_8"] = (
                df[f"{metric}_8"] - df[f"{metric}_{start_bar}"]
            )
    df["mfe_minus_mae_8"] = df["mfe_8"] - df["mae_8"]
    df["mfe_to_mae_8"] = df["mfe_8"] / df["mae_8"].replace(0, np.nan)
    df["mae_speed_1_8"] = df["mae_8"] / 8.0
    df["mae_speed_3_8"] = (df["mae_8"] - df["mae_3"]) / 5.0
    df["mae_speed_5_8"] = (df["mae_8"] - df["mae_5"]) / 3.0
    df["mfe_speed_1_8"] = df["mfe_8"] / 8.0
    df["mfe_speed_3_8"] = (df["mfe_8"] - df["mfe_3"]) / 5.0
    df["mfe_speed_5_8"] = (df["mfe_8"] - df["mfe_5"]) / 3.0
    for metric in ("mae", "mfe", "close"):
        columns = [f"{metric}_{bar}" for bar in range(1, S4_DECISION_BAR + 1)]
        df[f"{metric}_path_std_8"] = df[columns].std(axis=1)
    df["final_profitable"] = pd.to_numeric(df["net_R"], errors="coerce") > 0
    df["early_adverse"] = df["mae_8"] >= S4_ADVERSE_THRESHOLD_R
    adverse = df.loc[df["early_adverse"]].copy()
    adverse["recovery_group"] = np.where(
        adverse["final_profitable"], "RECOVERY", "FAILURE"
    )
    return adverse.reset_index(drop=True)


def run_s26_state_machine(row: pd.Series) -> dict[str, object]:
    benchmark_r = float(row["final_close_R"])
    mae_bars = sorted(
        int(column.split("_")[1][:-1])
        for column in row.index
        if column.startswith("mae_") and column.endswith("R")
    )
    close_bars = sorted(
        int(column.split("_")[1][:-1])
        for column in row.index
        if column.startswith("close_") and column.endswith("R")
    )
    mae_bar = next(
        (
            bar
            for bar in mae_bars
            if pd.notna(row.get(f"mae_{bar}R"))
            and float(row[f"mae_{bar}R"]) >= MAE_THRESHOLD_R
        ),
        None,
    )
    if mae_bar is None:
        return {
            "strategy_R": benchmark_r,
            "state": "NO_MAE_TRIGGER",
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "exit_bar": np.nan,
            "exit_type": "BENCHMARK",
        }

    deadline = min(mae_bar + RECOVERY_DEADLINE_BARS, max(close_bars))
    recovery_bar = next(
        (
            bar
            for bar in close_bars
            if mae_bar < bar <= deadline
            and pd.notna(row.get(f"close_{bar}R"))
            and float(row[f"close_{bar}R"]) >= RECOVERY_LEVEL_R
        ),
        None,
    )
    if recovery_bar is not None:
        return {
            "strategy_R": float(row[f"close_{recovery_bar}R"]),
            "state": "RECOVERED",
            "mae_bar": mae_bar,
            "recovery_bar": recovery_bar,
            "exit_bar": recovery_bar,
            "exit_type": "RECOVERY_EXIT",
        }

    failure_bar = deadline
    failure_value = row.get(f"close_{failure_bar}R")
    if pd.isna(failure_value):
        return {
            "strategy_R": benchmark_r,
            "state": "NO_EXECUTION_DATA",
            "mae_bar": mae_bar,
            "recovery_bar": np.nan,
            "exit_bar": np.nan,
            "exit_type": "BENCHMARK_FALLBACK",
        }
    return {
        "strategy_R": float(failure_value),
        "state": "FAILED_TO_RECOVER",
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "exit_bar": failure_bar,
        "exit_type": "FAILURE_EXIT",
    }


def run_s26(s4_cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for original_index, row in s4_cohort.reset_index(drop=True).iterrows():
        rows.append(
            {
                "original_index": original_index,
                "window": row["window"],
                "benchmark_R": float(row["final_close_R"]),
                **run_s26_state_machine(row),
            }
        )
    return pd.DataFrame(rows)


def trade_key(row: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Return the validated S2R economic identity.

    The reconstruction is strategy-scoped (S2R) and every S2 candidate is
    short. Identity is exactly entry timestamp, exit timestamp, and session
    ID. Window/model number is attribution metadata and is not part of the key.
    """
    entry = pd.to_datetime(row["entry_timestamp"], utc=True, errors="coerce")
    exit_time = pd.to_datetime(row["exit_timestamp"], utc=True, errors="coerce")
    if pd.isna(entry) or pd.isna(exit_time):
        raise ValueError("Invalid S2R trade timestamp.")
    return entry, exit_time, str(row["session_id"]).strip()


def integrate_s27(
    s2_trades: pd.DataFrame,
    s4_cohort: pd.DataFrame,
    s26_results: pd.DataFrame,
) -> pd.DataFrame:
    model: dict[tuple[pd.Timestamp, pd.Timestamp, str], dict[str, object]] = {}
    for _, result in s26_results.iterrows():
        index = int(result["original_index"])
        path = s4_cohort.iloc[index]
        key = trade_key(path)
        if key in model:
            raise RuntimeError("Duplicate S26/S4 trade identity.")
        model[key] = result.to_dict()

    rows = []
    seen: set[tuple[pd.Timestamp, pd.Timestamp, str]] = set()
    for _, trade in s2_trades.iterrows():
        key = trade_key(trade)
        if key in seen:
            raise RuntimeError("Duplicate S2 trade identity.")
        seen.add(key)
        result = model.get(key)
        matched = result is not None
        row = trade.to_dict()
        row.update(
            {
                "_s2_R": float(trade["net_R"]),
                "_model_match": "both" if matched else "left_only",
                "_model_strategy_R": result["strategy_R"] if matched else np.nan,
                "_model_state": result["state"] if matched else np.nan,
                "_model_mae_bar": result["mae_bar"] if matched else np.nan,
                "_model_recovery_bar": result["recovery_bar"] if matched else np.nan,
                "_model_exit_bar": result["exit_bar"] if matched else np.nan,
                "_model_exit_type": result["exit_type"] if matched else np.nan,
                "_strategy_R": (
                    float(result["strategy_R"]) if matched else float(trade["net_R"])
                ),
                "_state": result["state"] if matched else "NO_RECOVERY_ENRICHMENT",
                "_mae_bar": result["mae_bar"] if matched else np.nan,
                "_recovery_bar": result["recovery_bar"] if matched else np.nan,
                "_exit_bar": result["exit_bar"] if matched else np.nan,
                "_exit_type": result["exit_type"] if matched else "ORIGINAL_S2",
            }
        )
        row["_delta_R"] = row["_strategy_R"] - row["_s2_R"]
        rows.append(row)
    if len(seen) != len(s2_trades) or set(model).difference(seen):
        raise RuntimeError("S26/S4 model does not map one-to-one into S2 trades.")
    return pd.DataFrame(rows)


def reconstruct(
    market: pd.DataFrame,
    *,
    progress: bool = True,
) -> ReconstructionStages:
    rth = prepare_rth(market)
    windows = generate_windows(rth)
    if not windows:
        raise RuntimeError("No S2 walk-forward windows are available.")
    if progress:
        print(
            f"Raw rows={len(market):,}; RTH rows={len(rth):,}; "
            f"walk-forward windows={len(windows)}"
        )
        print("One-window real-data smoke check is the first full window below.")
    s2_trades, window_audit, s2_candidates = generate_s2_trades(
        rth, windows, progress=progress
    )
    if len(window_audit) != len(windows):
        raise RuntimeError(
            f"Processed {len(window_audit)} of {len(windows)} S2 windows."
        )
    s3_paths = build_s3_paths(market, s2_trades)
    s4_cohort = build_s4_cohort(s3_paths)
    s26_results = run_s26(s4_cohort)
    s27_trades = integrate_s27(s2_trades, s4_cohort, s26_results)
    return ReconstructionStages(
        market=market,
        window_audit=window_audit,
        s2_candidates=s2_candidates,
        s2_trades=s2_trades,
        s3_paths=s3_paths,
        s4_cohort=s4_cohort,
        s26_results=s26_results,
        s27_trades=s27_trades,
    )


def _normalized_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ("entry_timestamp", "exit_timestamp"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    result["session_id"] = result["session_id"].astype(str).str.strip()
    return result


def compare_artifact(
    label: str,
    rebuilt: pd.DataFrame,
    path: Path,
    value_columns: tuple[str, ...],
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> dict[str, object]:
    """Compare against saved research output; never feeds it into reconstruction."""

    reference = pd.read_csv(path)
    actual = _normalized_key_columns(rebuilt)
    expected = _normalized_key_columns(reference)
    if period is not None:
        actual = actual.loc[actual["entry_timestamp"].between(*period, inclusive="both")]
        expected = expected.loc[
            expected["entry_timestamp"].between(*period, inclusive="both")
        ]
    keys = ["entry_timestamp", "exit_timestamp", "session_id"]
    actual_keys = set(map(tuple, actual[keys].itertuples(index=False, name=None)))
    expected_keys = set(map(tuple, expected[keys].itertuples(index=False, name=None)))
    common = actual_keys & expected_keys
    actual_by_key = actual.set_index(keys, drop=False)
    expected_by_key = expected.set_index(keys, drop=False)
    mismatch_counts: dict[str, int] = {}
    value_mismatch_keys: set[tuple[object, ...]] = set()
    for column in value_columns:
        if column not in actual.columns or column not in expected.columns:
            mismatch_counts[column] = -1
            continue
        mismatches = 0
        for key in common:
            left = actual_by_key.loc[key, column]
            right = expected_by_key.loc[key, column]
            if pd.isna(left) and pd.isna(right):
                continue
            if isinstance(left, (int, float, np.number)) and isinstance(
                right, (int, float, np.number)
            ):
                if not np.isclose(float(left), float(right), rtol=0, atol=1e-10):
                    mismatches += 1
                    value_mismatch_keys.add(key)
            elif str(left) != str(right):
                mismatches += 1
                value_mismatch_keys.add(key)
        mismatch_counts[column] = mismatches
    all_mismatch_keys = (
        set(expected_keys - actual_keys)
        | set(actual_keys - expected_keys)
        | value_mismatch_keys
    )
    ordered_mismatch_keys = sorted(all_mismatch_keys)
    result = {
        "stage": label,
        "rebuilt_rows": len(actual),
        "artifact_rows": len(expected),
        "common_trade_keys": len(common),
        "missing_artifact_keys": len(expected_keys - actual_keys),
        "extra_rebuilt_keys": len(actual_keys - expected_keys),
        "value_mismatches": mismatch_counts,
        "first_mismatch": ordered_mismatch_keys[0] if ordered_mismatch_keys else None,
        "last_mismatch": ordered_mismatch_keys[-1] if ordered_mismatch_keys else None,
        "artifact": str(path),
    }
    print(f"{label} comparison: {result}")
    return result


def compare_with_frozen_artifacts(stages: ReconstructionStages) -> list[dict[str, object]]:
    comparisons = [
        compare_artifact(
            "S2",
            stages.s2_trades,
            S2_TRADES_PATH,
            (
                "quality",
                "vol_percentile",
                "raw_points",
                "net_points",
                "net_R",
                "exit_reason",
                "holding_bars",
                "window",
            ),
        ),
        compare_artifact(
            "S3",
            stages.s3_paths,
            S3_PATHS_PATH,
            ("entry_price", "path_length", "final_close_R", "max_MAE_R", "max_MFE_R"),
        ),
        compare_artifact(
            "S4",
            stages.s4_cohort,
            S4_COHORT_PATH,
            ("mae_8", "mfe_8", "close_8", "early_adverse", "recovery_group"),
        ),
    ]

    s26_reference = pd.read_csv(S26_RESULTS_PATH)
    s26_actual = stages.s26_results.copy()
    s26_expected = s26_reference.copy()
    s26_merged = s26_actual.merge(
        s26_expected,
        on="original_index",
        how="inner",
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    s26_values = (
        "window",
        "benchmark_R",
        "strategy_R",
        "state",
        "mae_bar",
        "recovery_bar",
        "exit_bar",
        "exit_type",
    )
    s26_mismatches: dict[str, int] = {}
    s26_value_mismatch_indices: set[int] = set()
    for column in s26_values:
        ref_column = f"{column}_reference"
        if column not in s26_merged or ref_column not in s26_merged:
            s26_mismatches[column] = -1
            continue
        left = s26_merged[column]
        right = s26_merged[ref_column]
        if column in {"state", "exit_type"}:
            equal = left.fillna("<NA>").astype(str) == right.fillna("<NA>").astype(str)
        else:
            left_num = pd.to_numeric(left, errors="coerce")
            right_num = pd.to_numeric(right, errors="coerce")
            both_null = left_num.isna() & right_num.isna()
            equal = both_null | np.isclose(
                left_num.fillna(0).to_numpy(dtype=float),
                right_num.fillna(0).to_numpy(dtype=float),
                rtol=0,
                atol=1e-10,
            )
        s26_mismatches[column] = int((~equal).sum())
        s26_value_mismatch_indices.update(
            s26_merged.loc[~equal, "original_index"].astype(int).tolist()
        )
    s26_key_mismatches = (
        set(s26_expected["original_index"].astype(int))
        .symmetric_difference(set(s26_actual["original_index"].astype(int)))
        | s26_value_mismatch_indices
    )
    s26_ordered_mismatches = sorted(s26_key_mismatches)
    s26_result = {
        "stage": "S26",
        "rebuilt_rows": len(s26_actual),
        "artifact_rows": len(s26_expected),
        "common_trade_keys": len(s26_merged),
        "missing_artifact_keys": int(
            (~s26_expected["original_index"].isin(s26_actual["original_index"])).sum()
        ),
        "extra_rebuilt_keys": int(
            (~s26_actual["original_index"].isin(s26_expected["original_index"])).sum()
        ),
        "value_mismatches": s26_mismatches,
        "first_mismatch": s26_ordered_mismatches[0] if s26_ordered_mismatches else None,
        "last_mismatch": s26_ordered_mismatches[-1] if s26_ordered_mismatches else None,
        "artifact": str(S26_RESULTS_PATH),
    }
    print(f"S26 comparison: {s26_result}")
    comparisons.append(s26_result)

    s27_columns = (
        "_strategy_R",
        "_state",
        "_mae_bar",
        "_recovery_bar",
        "_exit_bar",
        "_exit_type",
        "_model_match",
        "_delta_R",
    )
    comparisons.append(
        compare_artifact("S27", stages.s27_trades, S27_TRADES_PATH, s27_columns)
    )
    comparisons.append(
        compare_artifact(
            "S27 benchmark OOS 2020-06-23..2026-06-19",
            stages.s27_trades,
            S27_TRADES_PATH,
            s27_columns,
            period=(BENCHMARK_START, BENCHMARK_END),
        )
    )
    return comparisons


def report(stages: ReconstructionStages) -> None:
    print("\nS2 -> S3 -> S4 -> S26 -> S27 RAW RECONSTRUCTION")
    print(f"S2 walk-forward windows: {len(stages.window_audit)}")
    print(f"S2 signal-qualified candidates: {len(stages.s2_candidates)}")
    print(f"S2 trades: {len(stages.s2_trades)}")
    print(f"S3 path-enriched trades: {len(stages.s3_paths)}")
    print(f"S4 MAE_8 >= {S4_ADVERSE_THRESHOLD_R:.2f}R cohort: {len(stages.s4_cohort)}")
    print(f"S26 recovery outcomes: {len(stages.s26_results)}")
    print(f"S27 integrated trades: {len(stages.s27_trades)}")
    print("S26 states:")
    print(stages.s26_results["state"].value_counts().sort_index().to_string())
    entries = pd.to_datetime(stages.s27_trades["entry_timestamp"], utc=True)
    in_period = entries.between(BENCHMARK_START, BENCHMARK_END, inclusive="both")
    print(
        f"Benchmark entry period {BENCHMARK_START.date()}..{BENCHMARK_END.date()}: "
        f"{int(in_period.sum())} S2R trades"
    )
    candidate_timestamps = pd.to_datetime(stages.s2_candidates["timestamp"], utc=True)
    candidate_period = candidate_timestamps.between(
        BENCHMARK_START, BENCHMARK_END, inclusive="both"
    )
    print(
        "Signal-qualified candidates in benchmark entry period: "
        f"{int(candidate_period.sum())}"
    )
    holdout = stages.s27_trades.loc[
        pd.to_numeric(stages.s27_trades["window"], errors="coerce").between(12, 22)
    ]
    print(f"Research S29 holdout windows 12..22: {len(holdout)} trades")
    print(
        "S2R strategy_R total (all S27 rows): "
        f"{pd.to_numeric(stages.s27_trades['_strategy_R']).sum():.10f}"
    )


def validate_small_real_data(market: pd.DataFrame) -> None:
    sample_size = min(len(market), 5_000)
    if sample_size < 100:
        raise RuntimeError(f"Only {sample_size} raw market rows are available.")
    sample = add_directional_features(market.iloc[:sample_size].copy())
    required = (*BASE_FEATURES, *HMM_FEATURES)
    missing = [column for column in required if column not in sample]
    if missing:
        raise RuntimeError(f"Small real-data feature validation missing {missing}.")
    for column in required:
        left = pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(
            market.iloc[:sample_size][column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
            raise RuntimeError(
                f"Small real-data feature validation changed {column}."
            )
    print(
        f"Small real-data feature validation: PASS "
        f"({sample_size:,} Databento rows; feature values and row order unchanged)"
    )


def main() -> None:
    print("Loading raw Databento MNQ data and canonical features.")
    market = load_raw_feature_data()
    if market.empty:
        raise RuntimeError("Raw Databento MNQ loader returned no rows.")
    validate_small_real_data(market)
    stages = reconstruct(market)
    report(stages)
    print("\nComparing completed in-memory stages to saved research artifacts.")
    comparisons = compare_with_frozen_artifacts(stages)
    for result in comparisons:
        if (
            result["rebuilt_rows"] != result["artifact_rows"]
            or result["common_trade_keys"] != result["artifact_rows"]
            or result["missing_artifact_keys"]
            or result["extra_rebuilt_keys"]
            or any(count not in (0,) for count in result["value_mismatches"].values())
        ):
            raise RuntimeError(f"Frozen S2R comparison failed at {result['stage']}.")
    print("\nS2R RAW METHODOLOGICAL REPRODUCTION: PASS")


if __name__ == "__main__":
    main()
