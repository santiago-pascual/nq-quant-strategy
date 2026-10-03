from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data_loader import load_databento_mnq
from src.feature_engine import add_return_features, add_volatility_features
from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.paper.logger import PaperEventLogger, PaperEventType
from src.portfolio.conflict import PortfolioConflictEngine
from src.risk import RiskEngine
from src.risk.policy import XFA_50K_PRODUCTION_POLICY
from src.session_engine import add_session_information
from src.strategies.mean_reversion.config import (
    MRL1_CONFIG,
    MRS2_CONFIG,
    MeanReversionConfig,
)
from src.strategies.mean_reversion.strategy import MeanReversionStrategy
from src.strategies.orb.config import ORBConfig
from src.strategies.orb.strategy import ORBStrategy
from src.strategies.s2r.fitting import S2FittedModel, fit_s2_model
from src.strategies.s2r.signal import BASE_FEATURES
from src.strategies.s2r.strategy import S2RStrategy
from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.research.direction_features import add_directional_features
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "paper" / "research_replay"
DEFAULT_START = "2020-06-23"
DEFAULT_END = "2026-06-19"
OOS_STRATEGIES = ("MRL1", "MRS2", "S2R", "ORB")

FROZEN_INPUTS: dict[str, tuple[Path, str]] = {
    "hmm_states": (
        PROJECT_ROOT
        / "src"
        / "research"
        / "mean_reversion"
        / "results"
        / "cache"
        / "research_08b_causal_hmm_states.csv",
        "db036a030b6f787fb227163a8bca06a770c826d0bd100fcb0b525130e4b45b4e",
    ),
    "mean_reversion_trades": (
        PROJECT_ROOT
        / "src"
        / "research"
        / "mean_reversion"
        / "results"
        / "research_08aa_modular_reproduction_trades.csv",
        "72a9f09e3d4f47214e654f1d0d938c32c09bce40e6b719efa6586aa22ebd5c4a",
    ),
    "orb_trades": (
        PROJECT_ROOT
        / "src"
        / "research"
        / "results"
        / "orb"
        / "orb_reconciliation_trades.csv",
        "06af3494488771ceb09079249f0c4cc51fabaaff2a028cccf46d221ca51ed697",
    ),
    "s2r_trades": (
        PROJECT_ROOT
        / "src"
        / "research"
        / "results"
        / "s2_extended"
        / "s27_full_strategy_trades.csv",
        "4a17ee7d6d2c2083480c922907aad964993096d1dea6ece33317b3022ff5ddfb",
    ),
}

PARITY_FIELDS = (
    "entry_timestamp",
    "exit_timestamp",
    "direction",
    "entry_price",
    "exit_price",
    "exit_reason",
    "r_multiple",
    "candidate_id",
    "hmm_state",
    "window",
    "quality",
    "volatility_percentile",
    "volatility_percentile_scale",
    "zscore",
    "research_state",
    "research_exit_type",
    "session_date",
)
NUMERIC_PARITY_FIELDS = {
    "entry_price",
    "exit_price",
    "r_multiple",
    "quality",
    "volatility_percentile",
    "zscore",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen Research input is missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Frozen Research input hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def parse_timestamp_bound(value: str, *, end_of_day: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    is_date_only = len(value.strip()) == 10
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if end_of_day and is_date_only:
        timestamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def parse_oos_range(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_timestamp = parse_timestamp_bound(start)
    end_timestamp = parse_timestamp_bound(end, end_of_day=True)
    if end_timestamp < start_timestamp:
        raise ValueError("Replay end precedes replay start.")
    return start_timestamp, end_timestamp


def load_frozen_hmm_states(
    path: Path,
    *,
    expected_sha256: str,
) -> pd.DataFrame:
    verify_frozen_file(path, expected_sha256)
    states = pd.read_csv(path)
    required = {"timestamp", "hmm_state"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"Frozen HMM state file is missing columns: {sorted(missing)}")
    states["timestamp"] = pd.to_datetime(states["timestamp"], utc=True, errors="raise")
    states["hmm_state"] = pd.to_numeric(states["hmm_state"], errors="raise")
    if states["hmm_state"].isna().any() or not states["hmm_state"].isin((0, 1, 2)).all():
        raise ValueError("Frozen HMM states must be complete labels in 0..2.")
    states["hmm_state"] = states["hmm_state"].astype("int8")
    if "window" in states:
        states["research_hmm_window"] = pd.to_numeric(
            states["window"], errors="raise"
        ).astype("int16")
    else:
        states["research_hmm_window"] = pd.Series(
            pd.NA, index=states.index, dtype="Int16"
        )
    states = states[["timestamp", "hmm_state", "research_hmm_window"]]
    if states["timestamp"].duplicated().any():
        raise ValueError("Frozen HMM state file contains duplicate timestamps.")
    return states.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


class _FenwickTree:
    def __init__(self, size: int) -> None:
        self._tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int) -> None:
        while index < len(self._tree):
            self._tree[index] += 1
            index += index & -index

    def sum(self, index: int) -> int:
        result = 0
        while index:
            result += int(self._tree[index])
            index -= index & -index
        return result


def research_causal_expanding_percentile(values: pd.Series) -> pd.Series:
    """08AA percentile: rank current RV30 against strictly prior finite RTH rows."""
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(array), np.nan, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return pd.Series(result, index=values.index, name="vol_percentile")

    unique = np.unique(array[finite])
    tree = _FenwickTree(len(unique))
    previous_count = 0
    for row_index, value in enumerate(array):
        if not np.isfinite(value):
            continue
        rank = int(np.searchsorted(unique, value, side="right"))
        if previous_count:
            result[row_index] = tree.sum(rank) / previous_count
        insert_index = int(np.searchsorted(unique, value, side="left")) + 1
        tree.add(insert_index)
        previous_count += 1
    return pd.Series(result, index=values.index, name="vol_percentile")


@dataclass
class ResearchReplayContext:
    s2_models: dict[int, S2FittedModel]

    def s2_model_for_window(self, window: int) -> S2FittedModel:
        try:
            return self.s2_models[int(window)]
        except KeyError as exc:
            raise KeyError(f"No frozen Research S2R model for window {window}.") from exc


class ResearchReplayContextAdapter:
    """Pass Research-precomputed context through without causal inference."""

    def __init__(self, s2_models: Mapping[int, S2FittedModel] | None = None) -> None:
        self.context = ResearchReplayContext(dict(s2_models or {}))

    def update(self, market_data: Mapping[str, Any]) -> dict[str, Any]:
        return dict(market_data)


def _fit_s2r_models(
    market: pd.DataFrame,
    s2r_reference: pd.DataFrame,
) -> tuple[dict[int, S2FittedModel], pd.DataFrame]:
    schedule_columns = ["window", "validation_start", "validation_end"]
    missing = set(schedule_columns) - set(s2r_reference.columns)
    if missing:
        raise ValueError(f"S2R reference is missing schedule columns: {sorted(missing)}")
    schedule = s2r_reference[schedule_columns].drop_duplicates().copy()
    schedule["window"] = pd.to_numeric(schedule["window"], errors="raise").astype(int)
    schedule["validation_start"] = pd.to_datetime(
        schedule["validation_start"], utc=True, errors="raise"
    )
    schedule["validation_end"] = pd.to_datetime(
        schedule["validation_end"], utc=True, errors="raise"
    )
    schedule = schedule.sort_values("window", kind="mergesort").reset_index(drop=True)
    if schedule["window"].duplicated().any():
        raise ValueError("S2R reference has inconsistent metadata for a window.")
    if (schedule["validation_end"] < schedule["validation_start"]).any():
        raise ValueError("S2R reference has an inverted validation interval.")
    if (
        schedule["validation_start"].iloc[1:].reset_index(drop=True)
        < schedule["validation_end"].iloc[:-1].reset_index(drop=True)
    ).any():
        raise ValueError("S2R reference validation intervals overlap.")

    timestamps = market["timestamp"]
    windows = np.full(len(market), -1, dtype=np.int16)
    models: dict[int, S2FittedModel] = {}
    train_columns = ["hmm_state", *BASE_FEATURES, "realized_vol_30"]
    for schedule_index, item in enumerate(schedule.itertuples(index=False)):
        start = item.validation_start
        end = item.validation_end
        training_start = start - pd.DateOffset(years=2)
        inclusive_end = schedule_index == len(schedule) - 1
        validation_mask = (
            timestamps.ge(start)
            & (
                timestamps.le(end)
                if inclusive_end
                else timestamps.lt(end)
            )
        ).to_numpy()
        if (windows[validation_mask] >= 0).any():
            raise ValueError(f"S2R validation window {item.window} overlaps another.")
        windows[validation_mask] = int(item.window)

        train_mask = (
            timestamps.between(training_start, start, inclusive="both")
            & market["hmm_state"].notna()
        )
        train = market.loc[train_mask, train_columns].copy()
        if train.empty:
            raise ValueError(
                f"S2R window {item.window} has no frozen-state training rows."
            )
        models[int(item.window)] = fit_s2_model(train.to_dict(orient="records"))
    return models, pd.DataFrame(
        {"timestamp": timestamps, "s2r_window": windows}
    )


def _mark_final_orb_rth_bars(rth: pd.DataFrame) -> pd.Series:
    timestamps = rth["timestamp ET"]
    local_minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    config = ORBConfig()
    rth_start = config.rth_start_hour * 60 + config.rth_start_minute
    rth_end = config.rth_end_hour * 60 + config.rth_end_minute
    orb_session_bars = rth.loc[local_minutes.ge(rth_start) & local_minutes.lt(rth_end)]
    final_indices = orb_session_bars.groupby(
        orb_session_bars["timestamp ET"].dt.date,
        sort=False,
    ).tail(1).index
    return pd.Series(rth.index.isin(final_indices), index=rth.index, dtype=bool)


def prepare_research_market(
    raw: pd.DataFrame,
    frozen_states: pd.DataFrame,
    s2r_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, S2FittedModel]]:
    """Build frozen-context features and return only the Paper Engine bar fields."""
    required = {"timestamp ET", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Canonical MNQ data is missing columns: {sorted(missing)}")

    market = raw.copy()
    market["timestamp ET"] = pd.to_datetime(
        market["timestamp ET"], errors="raise"
    )
    market = market.sort_values("timestamp ET", kind="mergesort").reset_index(drop=True)
    market["timestamp"] = market["timestamp ET"].dt.tz_convert("UTC")
    if market["timestamp"].duplicated().any():
        raise ValueError("Canonical MNQ data contains duplicate timestamps.")

    market = add_session_information(market)
    market = add_return_features(market)
    market = add_volatility_features(market)
    market = add_directional_features(market)

    feature_columns = [
        "timestamp ET",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market_period",
        "realized_vol_30",
        *BASE_FEATURES,
    ]
    market = market[feature_columns].copy()
    rth_mask = market["market_period"].eq("RTH")
    rth = market.loc[rth_mask].copy()
    if rth.empty:
        raise ValueError("Canonical MNQ data contains no Research RTH rows.")

    # 08AA computes its z-score over RTH rows, while RV30/rank use the
    # canonical full-market RV30 series sampled at those same RTH timestamps.
    mr_features = build_mean_reversion_features(rth.copy())
    if "zscore_30" not in mr_features:
        raise ValueError("Canonical Mean Reversion features lack zscore_30.")
    rth["zscore"] = mr_features["zscore_30"].to_numpy(dtype=float)
    rth["vol_percentile"] = (
        research_causal_expanding_percentile(rth["realized_vol_30"]) * 100.0
    )

    rth = rth.merge(
        frozen_states,
        on="timestamp",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    models, window_map = _fit_s2r_models(rth, s2r_reference)
    rth = rth.merge(
        window_map,
        on="timestamp",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    rth = rth.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    rth["hmm_window"] = rth["s2r_window"].where(rth["s2r_window"] >= 0)
    rth["is_final_rth_bar"] = _mark_final_orb_rth_bars(rth)
    rth["timestamp"] = pd.to_datetime(rth["timestamp"], utc=True)
    # Preserve the exact OOS state label separately for audit; Paper Engine's
    # hmm_window is the S2R fit window consumed by its existing model hook.
    rth["hmm_state"] = pd.to_numeric(rth["hmm_state"], errors="coerce")
    rth["hmm_window"] = pd.to_numeric(rth["hmm_window"], errors="coerce")
    output_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hmm_state",
        "research_hmm_window",
        "hmm_window",
        "zscore",
        "vol_percentile",
        "realized_vol_30",
        *BASE_FEATURES,
        "is_final_rth_bar",
    ]
    return rth[output_columns].reset_index(drop=True), models


def load_verified_references() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    hashes: dict[str, str] = {}
    for name, (path, expected) in FROZEN_INPUTS.items():
        hashes[name] = verify_frozen_file(path, expected)
    frames = {
        name: pd.read_csv(path)
        for name, (path, _) in FROZEN_INPUTS.items()
        if name != "hmm_states"
    }
    return frames, hashes


def build_reference_trades(
    references: Mapping[str, pd.DataFrame],
    market: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    all_rth_timestamps = pd.DatetimeIndex(market["timestamp"])
    market_by_timestamp = market.set_index("timestamp", drop=False)
    closes = pd.Series(
        market["close"].to_numpy(dtype=float),
        index=all_rth_timestamps,
    )
    records: list[dict[str, Any]] = []

    def keep_in_range(timestamp: pd.Timestamp) -> bool:
        return bool(start <= timestamp <= end)

    mr = references["mean_reversion_trades"].copy()
    mr = mr.loc[mr["strategy_name"].isin(("MRL1", "MRS2"))]
    for row in mr.itertuples(index=False):
        entry = pd.Timestamp(row.entry_timestamp).tz_convert("UTC")
        if not keep_in_range(entry):
            continue
        entry_position = all_rth_timestamps.get_indexer([entry])[0]
        if entry_position < 0:
            raise ValueError(f"Mean Reversion entry does not map to an RTH bar: {entry}")
        exit_position = entry_position + int(row.bars_elapsed)
        if exit_position >= len(all_rth_timestamps):
            raise ValueError(f"Mean Reversion exit bar is outside the market data: {entry}")
        records.append(
            {
                "strategy_name": row.strategy_name,
                "entry_timestamp": entry,
                "exit_timestamp": all_rth_timestamps[exit_position],
                "direction": row.side,
                "entry_price": float(row.entry_price),
                "exit_price": float(row.exit_price),
                "exit_reason": _normalize_exit_reason(row.exit_reason),
                "r_multiple": float(row.r_multiple),
                "reference_candidate_id": row.candidate_id,
                "candidate_id": row.candidate_id,
                "hmm_state": int(row.entry_hmm_state),
                "window": int(row.window),
                "quality": np.nan,
                "volatility_percentile": float(
                    market_by_timestamp.at[entry, "vol_percentile"]
                ),
                "volatility_percentile_scale": "0..100_causal_expanding",
                "zscore": float(row.entry_zscore),
                "research_state": "",
                "research_exit_type": "",
                "session_date": "",
            }
        )

    s2r = references["s2r_trades"].copy()
    s2r_entry_column = "_entry_ts" if "_entry_ts" in s2r else "entry_timestamp"
    s2r_exit_column = "_exit_ts" if "_exit_ts" in s2r else "exit_timestamp"
    s2r[s2r_entry_column] = pd.to_datetime(s2r[s2r_entry_column], utc=True, errors="raise")
    s2r[s2r_exit_column] = pd.to_datetime(s2r[s2r_exit_column], utc=True, errors="raise")
    for row in s2r.to_dict(orient="records"):
        entry = pd.Timestamp(row[s2r_entry_column])
        if not keep_in_range(entry):
            continue
        exit_timestamp = pd.Timestamp(row[s2r_exit_column])
        entry_price = closes.get(entry, np.nan)
        if pd.isna(entry_price):
            raise ValueError(f"S2R entry does not map to an RTH bar: {entry}")
        raw_points = float(row["raw_points"])
        records.append(
            {
                "strategy_name": "S2R",
                "entry_timestamp": entry,
                "exit_timestamp": exit_timestamp,
                "direction": "SHORT",
                "entry_price": float(entry_price),
                "exit_price": float(entry_price - raw_points),
                "exit_reason": _normalize_exit_reason(row["exit_reason"]),
                "r_multiple": raw_points / float(row["stop_points"]),
                "reference_candidate_id": "S2R",
                "reference_net_r": float(row["net_R"]),
                "reference_quality": float(row["quality"]),
                "reference_vol_percentile": float(row["vol_percentile"]),
                "reference_hmm_state": 2,
                "reference_window": int(row["window"]),
                "candidate_id": "S2R",
                "hmm_state": 2,
                "window": int(row["window"]),
                "quality": float(row["quality"]),
                "volatility_percentile": float(row["vol_percentile"]),
                "volatility_percentile_scale": "0..1_fitted_reference",
                "zscore": np.nan,
                "research_state": row.get("_state", ""),
                "research_exit_type": row.get("_exit_type", ""),
                "session_date": str(row["session_id"]),
            }
        )

    orb = references["orb_trades"].copy()
    for row in orb.itertuples(index=False):
        entry = pd.Timestamp(row.entry_timestamp).tz_convert("UTC")
        if not keep_in_range(entry):
            continue
        raw_points = float(row.raw_points)
        direction = str(row.direction).upper()
        exit_price = float(row.entry_price) + (
            raw_points if direction == "LONG" else -raw_points
        )
        records.append(
            {
                "strategy_name": "ORB",
                "entry_timestamp": entry,
                "exit_timestamp": pd.Timestamp(row.exit_timestamp).tz_convert("UTC"),
                "direction": direction,
                "entry_price": float(row.entry_price),
                "exit_price": exit_price,
                "exit_reason": _normalize_exit_reason(row.exit_reason),
                "r_multiple": raw_points / float(row.risk_points),
                "reference_candidate_id": "ORB",
                "reference_risk_points": float(row.risk_points),
                "candidate_id": "ORB",
                "hmm_state": np.nan,
                "window": np.nan,
                "quality": np.nan,
                "volatility_percentile": np.nan,
                "volatility_percentile_scale": "",
                "zscore": np.nan,
                "research_state": "",
                "research_exit_type": "",
                "session_date": str(row.session_date),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(
            columns=["strategy_name", "entry_timestamp", "trade_key", *PARITY_FIELDS]
        )
    result["trade_key"] = result.apply(
        lambda row: _trade_key(row["strategy_name"], row["entry_timestamp"]), axis=1
    )
    if result["trade_key"].duplicated().any():
        raise ValueError("Frozen trade artifacts contain duplicate strategy entries.")
    return result.sort_values(
        ["strategy_name", "entry_timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_exit_reason(reason: object) -> str:
    text = str(reason).strip().lower().replace("-", "_").replace(" ", "_")
    if "failed_to_recover" in text:
        return "failed_to_recover"
    if "recover" in text:
        return "recovered"
    if "target" in text or "take_profit" in text:
        return "target"
    if "stop" in text:
        return "stop"
    if "timeout" in text:
        return "timeout"
    if "rth_close" in text:
        return "rth_close"
    return text


def _trade_key(strategy_name: object, timestamp: object) -> str:
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        raise ValueError("Trade keys require timezone-aware timestamps.")
    return f"{str(strategy_name).upper()}|{stamp.tz_convert('UTC').isoformat()}"


class _ReplayEventLogger(PaperEventLogger):
    """Retain only lifecycle and attribution events, not one file record per bar."""

    _CAPTURED = {
        PaperEventType.STRATEGY_DECISION,
        PaperEventType.RISK_REQUEST,
        PaperEventType.FILL,
        PaperEventType.POSITION_OPENED,
        PaperEventType.POSITION_CLOSED,
    }

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(
        self,
        event_type: PaperEventType,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime | None = None,
    ) -> None:
        if event_type not in self._CAPTURED:
            return
        if (
            event_type is PaperEventType.STRATEGY_DECISION
            and payload.get("action") not in {"enter", "exit"}
        ):
            return
        if timestamp is None or timestamp.tzinfo is None:
            raise ValueError("Paper Engine lifecycle event timestamp must be aware.")
        self.events.append(
            {
                "event_type": event_type.value,
                "timestamp": pd.Timestamp(timestamp).tz_convert("UTC"),
                "payload": dict(payload),
            }
        )


def _build_paper_engine(
    strategies: Sequence[str],
    *,
    context_adapter: ResearchReplayContextAdapter,
    initial_equity: float,
    commission_per_contract: float,
    price_offset: float,
    logger: _ReplayEventLogger,
) -> PaperTradingEngine:
    configs: dict[str, MeanReversionConfig] = {
        "MRL1": MRL1_CONFIG,
        "MRS2": MRS2_CONFIG,
    }
    strategy_objects = []
    for name in strategies:
        if name in configs:
            strategy_objects.append(MeanReversionStrategy(configs[name]))
        elif name == "S2R":
            strategy_objects.append(S2RStrategy())
        elif name == "ORB":
            strategy_objects.append(ORBStrategy())
        else:
            raise ValueError(f"Unsupported Research Replay strategy: {name}")
    return PaperTradingEngine(
        strategies=strategy_objects,
        execution=ExecutionEngine(),
        risk=RiskEngine(XFA_50K_PRODUCTION_POLICY.to_risk_limits()),
        conflict=PortfolioConflictEngine(max_concurrent_positions=3),
        broker=InMemoryBrokerAdapter(),
        logger=logger,
        context_adapter=context_adapter,
        config=PaperEngineConfig(
            initial_equity=initial_equity,
            point_value=2.0,
            tick_size=0.25,
            price_offset=price_offset,
            commission_per_contract=commission_per_contract,
            automatic_simulated_fills=True,
        ),
    )


def _paper_trades_from_events(
    events: Iterable[Mapping[str, Any]],
    *,
    point_value: float,
    commission_per_contract: float,
) -> pd.DataFrame:
    pending_entry: dict[str, dict[str, Any]] = {}
    pending_exit_reason: dict[str, str] = {}
    pending_exit_detail: dict[str, str] = {}
    pending_risk: dict[str, dict[str, Any]] = {}
    pending_commission: dict[str, float] = {}
    active: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []

    for event in events:
        kind = event["event_type"]
        timestamp = pd.Timestamp(event["timestamp"])
        payload = event["payload"]
        strategy = str(payload.get("strategy_name", ""))
        if kind == PaperEventType.STRATEGY_DECISION.value:
            action = payload.get("action")
            if action == "enter":
                pending_entry[strategy] = {
                    "entry_signal_timestamp": timestamp,
                    "entry_signal_reason": payload.get("reason"),
                }
            elif action == "exit":
                pending_exit_detail[strategy] = str(payload.get("reason", ""))
                pending_exit_reason[strategy] = _normalize_exit_reason(
                    pending_exit_detail[strategy]
                )
        elif kind == PaperEventType.RISK_REQUEST.value:
            pending_risk[strategy] = dict(payload)
        elif kind == PaperEventType.FILL.value:
            fee = int(payload.get("quantity", 0)) * commission_per_contract
            if strategy in active:
                active[strategy]["commission"] += fee
            else:
                pending_commission[strategy] = pending_commission.get(strategy, 0.0) + fee
        elif kind == PaperEventType.POSITION_OPENED.value:
            active[strategy] = {
                **payload,
                **pending_entry.pop(strategy, {}),
                "entry_timestamp": timestamp,
                "commission": pending_commission.pop(strategy, 0.0),
                "risk_request": pending_risk.pop(strategy, {}),
            }
        elif kind == PaperEventType.POSITION_CLOSED.value:
            opened = active.pop(strategy, None)
            if opened is None:
                raise RuntimeError(f"Paper Engine closed {strategy} without an open record.")
            direction = str(payload.get("side", opened.get("side", ""))).upper()
            entry_price = float(payload["entry_price"])
            exit_price = float(payload["exit_price"])
            quantity = int(payload.get("quantity", opened.get("quantity", 0)))
            signed_points = (
                exit_price - entry_price
                if direction == "LONG"
                else entry_price - exit_price
            )
            gross_pnl = signed_points * quantity * point_value
            commission = float(opened["commission"])
            risk_request = opened["risk_request"]
            risk_points = abs(
                float(risk_request.get("entry_price", entry_price))
                - float(risk_request.get("stop_price", entry_price))
            )
            if risk_points <= 0:
                raise ValueError(f"Paper Engine trade {strategy} has no positive risk.")
            signal_timestamp = opened.get("entry_signal_timestamp", opened["entry_timestamp"])
            trades.append(
                {
                    "strategy_name": strategy,
                    "trade_key": _trade_key(strategy, signal_timestamp),
                    "entry_signal_timestamp": signal_timestamp,
                    "entry_signal_reason": opened.get("entry_signal_reason"),
                    "entry_timestamp": opened["entry_timestamp"],
                    "exit_timestamp": timestamp,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": pending_exit_reason.pop(strategy, ""),
                    "exit_reason_detail": pending_exit_detail.pop(strategy, ""),
                    "quantity": quantity,
                    "risk_points": risk_points,
                    "gross_points": signed_points,
                    "r_multiple": signed_points / risk_points,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "net_pnl": gross_pnl - commission,
                    "net_r_multiple": (gross_pnl - commission)
                    / (risk_points * point_value * quantity),
                }
            )
    return pd.DataFrame(trades)


def compare_trade_ledgers(
    reference: pd.DataFrame,
    paper: pd.DataFrame,
    *,
    absolute_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Compare trade-level identity and execution fields, reporting first divergence."""
    for name, frame in (("reference", reference), ("paper", paper)):
        if "trade_key" not in frame:
            raise ValueError(f"{name} ledger requires trade_key.")
        if frame["trade_key"].duplicated().any():
            raise ValueError(f"{name} ledger contains duplicate trade keys.")

    ref_rows = reference.set_index("trade_key", drop=False).to_dict(orient="index")
    paper_rows = paper.set_index("trade_key", drop=False).to_dict(orient="index")
    keys = sorted(set(ref_rows) | set(paper_rows))
    rows: list[dict[str, Any]] = []
    for key in keys:
        expected = ref_rows.get(key)
        actual = paper_rows.get(key)
        status = "MATCH"
        first_difference = ""
        if expected is None:
            status = "EXTRA"
        elif actual is None:
            status = "MISSING"
        else:
            for field in PARITY_FIELDS:
                left = expected.get(field)
                right = actual.get(field)
                if field in NUMERIC_PARITY_FIELDS:
                    equal = (
                        pd.isna(left)
                        and pd.isna(right)
                    ) or (
                        pd.notna(left)
                        and pd.notna(right)
                        and np.isclose(
                            float(left),
                            float(right),
                            rtol=0,
                            atol=absolute_tolerance,
                        )
                    )
                elif field.endswith("_timestamp"):
                    equal = (
                        pd.isna(left)
                        and pd.isna(right)
                    ) or (
                        pd.notna(left)
                        and pd.notna(right)
                        and pd.Timestamp(left).tz_convert("UTC")
                        == pd.Timestamp(right).tz_convert("UTC")
                    )
                elif field in {"hmm_state", "window"}:
                    equal = (
                        pd.isna(left) and pd.isna(right)
                    ) or (
                        pd.notna(left)
                        and pd.notna(right)
                        and int(left) == int(right)
                    )
                else:
                    equal = (
                        pd.isna(left) and pd.isna(right)
                    ) or str(left).upper() == str(right).upper()
                if not equal:
                    status = "MISMATCH"
                    first_difference = field
                    break
        row: dict[str, Any] = {
            "trade_key": key,
            "strategy_name": (expected or actual or {}).get("strategy_name", ""),
            "status": status,
            "first_difference_field": first_difference,
        }
        for field in PARITY_FIELDS:
            row[f"reference_{field}"] = (expected or {}).get(field)
            row[f"paper_{field}"] = (actual or {}).get(field)
        rows.append(row)
    columns = [
        "trade_key",
        "strategy_name",
        "status",
        "first_difference_field",
        *[f"reference_{field}" for field in PARITY_FIELDS],
        *[f"paper_{field}" for field in PARITY_FIELDS],
    ]
    return pd.DataFrame(rows, columns=columns)


def add_paper_trade_attribution(
    paper: pd.DataFrame,
    market: pd.DataFrame,
    s2_models: Mapping[int, S2FittedModel],
) -> pd.DataFrame:
    if paper.empty:
        return paper
    market_by_timestamp = market.set_index("timestamp", drop=False)
    enriched = paper.copy()
    attribution: list[dict[str, Any]] = []
    for trade in enriched.to_dict(orient="records"):
        signal_timestamp = pd.Timestamp(trade["entry_signal_timestamp"])
        context = market_by_timestamp.loc[signal_timestamp]
        strategy = trade["strategy_name"]
        record: dict[str, Any] = {
            "candidate_id": {
                "MRL1": MRL1_CONFIG.candidate_id,
                "MRS2": MRS2_CONFIG.candidate_id,
                "S2R": "S2R",
                "ORB": "ORB",
            }[strategy],
            "hmm_state": (
                int(context["hmm_state"])
                if strategy != "ORB" and pd.notna(context["hmm_state"])
                else np.nan
            ),
            "window": np.nan,
            "quality": np.nan,
            "volatility_percentile": np.nan,
            "volatility_percentile_scale": "",
            "zscore": (
                float(context["zscore"])
                if strategy in {"MRL1", "MRS2"} and pd.notna(context["zscore"])
                else np.nan
            ),
            "research_state": "",
            "research_exit_type": "",
            "session_date": "",
        }
        if strategy in {"MRL1", "MRS2"}:
            record["window"] = (
                int(context["research_hmm_window"])
                if pd.notna(context["research_hmm_window"])
                else np.nan
            )
            record["volatility_percentile"] = (
                float(context["vol_percentile"])
                if pd.notna(context["vol_percentile"])
                else np.nan
            )
            record["volatility_percentile_scale"] = "0..100_causal_expanding"
        elif strategy == "S2R":
            window = int(context["hmm_window"])
            fitted = s2_models[window]
            values = {feature: float(context[feature]) for feature in BASE_FEATURES}
            record["window"] = window
            record["quality"] = fitted.signal_model.calculate_quality(values)
            record["volatility_percentile"] = fitted.transform_volatility(
                float(context["realized_vol_30"])
            )
            record["volatility_percentile_scale"] = "0..1_fitted_reference"
            detail = str(trade.get("exit_reason_detail", "")).lower()
            if "failed_to_recover" in detail:
                record["research_state"] = "FAILED_TO_RECOVER"
                record["research_exit_type"] = "RECOVERY"
            elif "recovered" in detail:
                record["research_state"] = "RECOVERED"
                record["research_exit_type"] = "RECOVERY"
            else:
                record["research_state"] = "NO_RECOVERY_ENRICHMENT"
                record["research_exit_type"] = "ORIGINAL_S2"
            record["session_date"] = str(
                (
                    pd.Timestamp(context["timestamp"])
                    .tz_convert("America/New_York")
                    - pd.Timedelta(hours=18)
                ).date()
            )
        else:
            record["session_date"] = str(
                pd.Timestamp(context["timestamp"])
                .tz_convert("America/New_York")
                .date()
            )
        attribution.append(record)
    attributes = pd.DataFrame(attribution, index=enriched.index)
    for column in attributes:
        enriched[column] = attributes[column]
    return enriched


def summarize_replay(
    paper: pd.DataFrame,
    parity: pd.DataFrame,
    strategies: Sequence[str],
    *,
    open_positions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        strategy_parity = (
            parity.loc[parity["strategy_name"].eq(strategy)]
            if "strategy_name" in parity
            else parity
        )
        strategy_paper = (
            paper.loc[paper["strategy_name"].eq(strategy)]
            if "strategy_name" in paper
            else paper
        )
        rows.append(
            {
                "strategy_name": strategy,
                "reference_trades": int(
                    strategy_parity["status"].isin(("MATCH", "MISMATCH", "MISSING")).sum()
                ),
                "paper_closed_trades": int(len(strategy_paper)),
                "exact_matches": int(strategy_parity["status"].eq("MATCH").sum()),
                "mismatches": int(strategy_parity["status"].eq("MISMATCH").sum()),
                "missing": int(strategy_parity["status"].eq("MISSING").sum()),
                "extra": int(strategy_parity["status"].eq("EXTRA").sum()),
                "gross_pnl": float(strategy_paper["gross_pnl"].sum())
                if not strategy_paper.empty
                else 0.0,
                "commissions": float(strategy_paper["commission"].sum())
                if not strategy_paper.empty
                else 0.0,
                "net_pnl": float(strategy_paper["net_pnl"].sum())
                if not strategy_paper.empty
                else 0.0,
                "open_positions_at_end": int(open_positions),
            }
        )
    return pd.DataFrame(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay frozen Research context through the current Paper Engine."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="UTC date or timestamp.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive UTC date or timestamp.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", choices=OOS_STRATEGIES)
    parser.add_argument("--initial-equity", type=float, default=50_000.0)
    parser.add_argument("--commission-per-contract", type=float, default=0.0)
    parser.add_argument("--price-offset", type=float, default=0.0)
    return parser


def run_research_replay(args: argparse.Namespace) -> dict[str, Path]:
    start, end = parse_oos_range(args.start, args.end)
    selected = (args.strategy,) if args.strategy else OOS_STRATEGIES
    references, hashes = load_verified_references()
    frozen_states = load_frozen_hmm_states(
        FROZEN_INPUTS["hmm_states"][0],
        expected_sha256=FROZEN_INPUTS["hmm_states"][1],
    )
    raw = load_databento_mnq()
    market, s2_models = prepare_research_market(
        raw,
        frozen_states,
        references["s2r_trades"],
    )
    timestamps = pd.DatetimeIndex(market["timestamp"])
    replay_start = int(timestamps.searchsorted(start, side="left"))
    oos_end = int(timestamps.searchsorted(end, side="right"))
    if replay_start >= oos_end:
        raise ValueError(f"No canonical Research RTH bars in {start}..{end}.")
    # Continue the Paper Engine through the longest current strategy path so
    # end-of-range entries are compared after their lifecycle has resolved.
    replay_end = min(oos_end + 40, len(market))
    replay_market = market.iloc[replay_start:replay_end].copy()
    reference_trades = build_reference_trades(references, market, start, end)
    reference_trades = reference_trades.loc[
        reference_trades["strategy_name"].isin(selected)
    ].reset_index(drop=True)

    context_adapter = ResearchReplayContextAdapter(s2_models)
    logger = _ReplayEventLogger()
    engine = _build_paper_engine(
        selected,
        context_adapter=context_adapter,
        initial_equity=args.initial_equity,
        commission_per_contract=args.commission_per_contract,
        price_offset=args.price_offset,
        logger=logger,
    )
    engine.connect()
    try:
        for row in replay_market.itertuples(index=False):
            state = None if pd.isna(row.hmm_state) else int(row.hmm_state)
            window = None if pd.isna(row.hmm_window) else int(row.hmm_window)
            market_data = {
                "timestamp": row.timestamp.to_pydatetime(),
                "symbol": "MNQ",
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
                "hmm_state": state,
                "research_hmm_window": (
                    None
                    if pd.isna(row.research_hmm_window)
                    else int(row.research_hmm_window)
                ),
                "hmm_window": window,
                "zscore": None if pd.isna(row.zscore) else float(row.zscore),
                "vol_percentile": (
                    None if pd.isna(row.vol_percentile) else float(row.vol_percentile)
                ),
                "realized_vol_30": (
                    None
                    if pd.isna(row.realized_vol_30)
                    else float(row.realized_vol_30)
                ),
                "is_final_rth_bar": bool(row.is_final_rth_bar),
            }
            market_data.update(
                {
                    feature: (
                        None if pd.isna(getattr(row, feature)) else float(getattr(row, feature))
                    )
                    for feature in BASE_FEATURES
                }
            )
            engine.process_bar(market_data)
    finally:
        engine.disconnect()

    paper_trades = _paper_trades_from_events(
        logger.events,
        point_value=engine.config.point_value,
        commission_per_contract=engine.config.commission_per_contract,
    )
    if not paper_trades.empty:
        paper_trades = paper_trades.loc[
            paper_trades["entry_signal_timestamp"].between(start, end, inclusive="both")
        ].reset_index(drop=True)
        paper_trades = add_paper_trade_attribution(
            paper_trades,
            market,
            s2_models,
        )
    parity = compare_trade_ledgers(reference_trades, paper_trades)
    summary = summarize_replay(
        paper_trades,
        parity,
        selected,
        open_positions=len(engine.execution.get_positions()),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.csv",
        "trades": output_dir / "trades.csv",
        "parity": output_dir / "parity.csv",
        "run_summary": output_dir / "run_summary.json",
    }
    summary.to_csv(paths["summary"], index=False)
    paper_trades.to_csv(paths["trades"], index=False)
    parity.to_csv(paths["parity"], index=False)
    run_summary = {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "strategies": list(selected),
        "official_oos_rth_bars": int(oos_end - replay_start),
        "replay_rth_bars_including_exit_buffer": int(len(replay_market)),
        "post_end_exit_buffer_rth_bars": int(replay_end - oos_end),
        "frozen_inputs_sha256": hashes,
        "output_files": {name: path.name for name, path in paths.items()},
        "initial_equity": args.initial_equity,
        "commission_per_contract": args.commission_per_contract,
        "price_offset_points": args.price_offset,
        "causal_inference_used": False,
        "open_positions_at_end": len(engine.execution.get_positions()),
    }
    paths["run_summary"].write_text(
        json.dumps(run_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    paths = run_research_replay(args)
    print(f"Research Replay outputs: {Path(args.output_dir).resolve()}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
