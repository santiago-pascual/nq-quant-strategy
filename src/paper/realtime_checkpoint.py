"""Atomic durable runtime checkpoints for the Paper-only engine."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd


CHECKPOINT_SCHEMA_VERSION = 1


def runtime_identity() -> dict[str, Any]:
    """Versioned execution and reporting identity used for Paper recovery."""
    from src.paper.runtime_compatibility import (
        IDENTITY_SCHEMA_VERSION,
        COMPATIBILITY_POLICY_VERSION,
        REPORTING_SCHEMA_VERSION,
        compatibility_policy_sha256,
    )
    root = Path(__file__).resolve().parents[2]
    source_paths = (
        "src/models/causal_hmm.py",
        "src/paper/market_context.py",
        "src/paper/context_adapter.py",
        "src/paper/engine.py",
        "src/paper/run_autonomous.py",
        "src/paper/realtime_service.py",
        "src/paper/realtime_checkpoint.py",
        "src/paper/delayed_paper_cli.py",
        "src/paper/logger.py",
        "src/paper/realtime_market_data.py",
        "src/paper/run_realtime_paper.py",
        "src/paper/shadow_replay.py",
        "src/paper/cme_calendar.py",
        "src/paper/costs.py",
        "src/paper/analytics.py",
        "src/paper/analytics_db.py",
        "src/paper/ibkr_paper_recovery.py",
        "src/execution/engine.py",
        "src/broker/adapter.py",
        "src/portfolio/conflict.py",
        "src/strategies/mean_reversion/strategy.py",
        "src/strategies/orb/strategy.py",
        "src/strategies/s2r/strategy.py",
        "src/risk/engine.py",
        "src/risk/policy.py",
    )
    files = {}
    for relative in source_paths:
        path = root / relative
        files[relative] = sha256(path.read_bytes()).hexdigest()
    files["src/paper/runtime_compatibility.py"] = compatibility_policy_sha256()
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "hmmlearn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    execution_sources = {key: value for key, value in files.items()
                         if key != "src/paper/analytics.py"}
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "compatibility_policy_version": COMPATIBILITY_POLICY_VERSION,
        "python": platform.python_version(),
        "packages": packages,
        "source_sha256": files,
        "execution_fingerprint": {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "source_sha256": execution_sources,
        },
        "reporting_fingerprint": {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "source_sha256": {"src/paper/analytics.py": files["src/paper/analytics.py"]},
        },
    }


def _encode(value: Any) -> Any:
    # str-backed Enum members (for example RiskDecision) are also isinstance
    # checks for str. Preserve their type so restored execution decisions keep
    # their enum semantics rather than becoming bare strings.
    if isinstance(value, Enum):
        return {"$enum": f"{type(value).__module__}:{type(value).__qualname__}", "value": value.value}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, pd.Timestamp):
        return {"$type": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"$type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"$type": "date", "value": value.isoformat()}
    if is_dataclass(value):
        return {
            "$dataclass": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, set):
        # Set iteration order is process/hash-seed dependent. Sort by a stable
        # JSON representation so equivalent restored execution state produces
        # byte-stable checkpoint payloads across restarts.
        encoded = [_encode(item) for item in value]
        return sorted(encoded, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=True
        ))
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_encode(item) for item in value]
    raise TypeError(f"Unsupported runtime checkpoint value: {type(value).__name__}")


def _class_for(tag: str):
    module_name, separator, qualname = tag.partition(":")
    if not separator or not module_name.startswith("src."):
        raise ValueError("Checkpoint contains an unapproved type reference")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$type" in value:
        constructors = {
            "timestamp": pd.Timestamp,
            "datetime": datetime.fromisoformat,
            "date": date.fromisoformat,
        }
        kind = value["$type"]
        if kind not in constructors:
            raise ValueError(f"Unknown checkpoint scalar type: {kind}")
        return constructors[kind](value["value"])
    if "$enum" in value:
        return _class_for(value["$enum"])(value["value"])
    if "$dataclass" in value:
        cls = _class_for(value["$dataclass"])
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    return {key: _decode(item) for key, item in value.items()}


def engine_fingerprint(engine: Any) -> str:
    from dataclasses import asdict as dataclass_asdict

    config = {
        "engine": dataclass_asdict(engine.config),
        "execution": {
            "deterministic_ids": bool(getattr(engine.execution, "deterministic_ids", False)),
        },
        "risk": dataclass_asdict(engine.risk.limits),
        "strategies": [
            {
                "name": strategy.name,
                "version": strategy.version,
                "config": dataclass_asdict(strategy.config)
                if hasattr(strategy, "config") and is_dataclass(strategy.config)
                else None,
            }
            for strategy in engine.strategies
        ],
    }
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(serialized.encode()).hexdigest()


def capture_engine_state(engine: Any) -> dict[str, Any]:
    """Capture completed-bar state, including unsettled in-memory Paper orders."""
    execution = engine.execution
    broker = engine.broker
    if not all(hasattr(broker, name) for name in ("_orders", "_fills", "_positions", "_order_counter")):
        raise TypeError("Realtime Paper checkpoints currently require the in-memory Paper broker")
    if not hasattr(engine.broker_execution, "_broker_orders"):
        raise TypeError("Realtime Paper checkpoints require the in-memory broker execution coordinator")
    strategies: dict[str, Any] = {}
    for strategy in engine.strategies:
        if strategy.name in {"MRL1", "MRS2"}:
            strategies[strategy.name] = {
                "kind": "mean_reversion",
                "trade_state": strategy._trade_state,
                "pending_exit_price": strategy._pending_exit_price,
            }
        elif strategy.name == "ORB":
            builder = strategy._context_builder
            strategies[strategy.name] = {
                "kind": "orb",
                "context": strategy._context,
                "builder": {
                    "session_date": builder._session_date,
                    "or_high": builder._or_high,
                    "or_low": builder._or_low,
                    "or_bars": builder._or_bars,
                    "trade_taken": builder._trade_taken,
                    "session_invalidated": builder._session_invalidated,
                },
                "trade_state": strategy._trade_state,
                "pending_or_high": strategy._pending_or_high,
                "pending_or_low": strategy._pending_or_low,
                "pending_exit_price": strategy._pending_exit_price,
                "in_trade": strategy._in_trade,
            }
        elif strategy.name == "S2R":
            recovery = strategy._recovery
            strategies[strategy.name] = {
                "kind": "s2r",
                "in_trade": strategy._in_trade,
                "entry_price": strategy._entry_price,
                "entry_bar": strategy._entry_bar,
                "last_bar_index": strategy._last_bar_index,
                "pending_exit_price": strategy._pending_exit_price,
                "recovery": {
                    "state": recovery.state,
                    "mae_bar": recovery.mae_bar,
                    "recovery_bar": recovery.recovery_bar,
                    "exit_bar": recovery.exit_bar,
                    "last_bar": recovery._last_bar,
                },
                "fitted_model": strategy.fitted_model,
                "model_window": strategy.model_window,
            }
        else:
            raise TypeError(f"No checkpoint codec for strategy {strategy.name}")
    return _encode({
        "fingerprint": engine_fingerprint(engine),
        "last_timestamp": engine._last_timestamp,
        "last_market_data": engine._last_market_data,
        "last_decisions": engine._last_decisions,
        "realized_pnl": engine._realized_pnl,
        "commissions": engine._commissions,
        "exchange_fees": engine._exchange_fees,
        "regulatory_fees": engine._regulatory_fees,
        "account_equity": engine._account_equity,
        "position_commissions": engine._position_commissions,
        "position_exchange_fees": engine._position_exchange_fees,
        "position_regulatory_fees": engine._position_regulatory_fees,
        "position_initial_risk": engine._position_initial_risk,
        "execution": {
            "orders": execution._orders,
            "positions": execution._positions,
            "closed_positions": execution._closed_positions,
            "processed_fill_ids": execution._processed_fill_ids,
            "processed_intents": execution._processed_intents,
            "order_intents": execution._order_intents,
        },
        "risk": {
            "open_positions": engine.risk._open_positions,
            "daily_trade_count": engine.risk._daily_trade_count,
            "daily_realized_pnl": engine.risk._daily_realized_pnl,
            "current_day": engine.risk._current_day,
        },
        "conflict": engine.conflict._positions,
        "broker": {
            "orders": broker._orders,
            "fills": broker._fills,
            "positions": broker._positions,
            "order_counter": broker._order_counter,
            "fill_counter": broker._fill_counter,
        },
        "pending_execution": {
            "risk_results": engine._pending_risk_results,
            "entry_requests": engine._pending_entry_requests,
            "simulated_orders": engine._simulated_orders,
            "strategy_orders": engine._pending_strategy_orders,
            "broker_submissions": engine.broker_execution._broker_orders,
            "newly_filled_strategies": engine._newly_filled_strategies,
        },
        "strategies": strategies,
    })


def restore_engine_state(engine: Any, encoded_state: Mapping[str, Any]) -> None:
    state = _decode(dict(encoded_state))
    if state["fingerprint"] != engine_fingerprint(engine):
        raise ValueError("Paper engine configuration differs from checkpoint")
    engine._last_timestamp = state["last_timestamp"]
    engine._last_market_data = state["last_market_data"]
    engine._last_decisions = state["last_decisions"]
    engine._realized_pnl = float(state["realized_pnl"])
    engine._commissions = float(state["commissions"])
    engine._exchange_fees = float(state.get("exchange_fees", 0.0))
    engine._regulatory_fees = float(state.get("regulatory_fees", 0.0))
    engine._account_equity = float(state["account_equity"])
    engine._position_commissions = dict(state["position_commissions"])
    engine._position_exchange_fees = dict(state.get("position_exchange_fees", {}))
    engine._position_regulatory_fees = dict(state.get("position_regulatory_fees", {}))
    engine._position_initial_risk = dict(state.get("position_initial_risk", {}))
    execution = engine.execution
    execution._orders = state["execution"]["orders"]
    execution._positions = state["execution"]["positions"]
    execution._closed_positions = state["execution"]["closed_positions"]
    execution._processed_fill_ids = set(state["execution"]["processed_fill_ids"])
    execution._processed_intents = set(state["execution"]["processed_intents"])
    execution._order_intents = state["execution"]["order_intents"]
    risk = engine.risk
    risk._open_positions = state["risk"]["open_positions"]
    risk._daily_trade_count = int(state["risk"]["daily_trade_count"])
    risk._daily_realized_pnl = float(state["risk"]["daily_realized_pnl"])
    risk._current_day = state["risk"]["current_day"]
    engine.conflict._positions = state["conflict"]
    broker = engine.broker
    broker._orders = state["broker"]["orders"]
    broker._fills = state["broker"]["fills"]
    broker._positions = state["broker"]["positions"]
    broker._order_counter = int(state["broker"]["order_counter"])
    broker._fill_counter = int(state["broker"]["fill_counter"])
    pending = state.get("pending_execution", {})
    engine._pending_risk_results = dict(pending.get("risk_results", {}))
    engine._pending_entry_requests = dict(pending.get("entry_requests", {}))
    engine._simulated_orders = dict(pending.get("simulated_orders", {}))
    engine._pending_strategy_orders = {
        name: set(order_ids) for name, order_ids in pending.get("strategy_orders", {}).items()
    }
    engine.broker_execution._broker_orders = dict(pending.get("broker_submissions", {}))
    engine._newly_filled_strategies = set(pending.get("newly_filled_strategies", []))
    _validate_pending_execution(engine)
    for strategy in engine.strategies:
        saved = state["strategies"][strategy.name]
        if saved["kind"] == "mean_reversion":
            strategy._trade_state = saved["trade_state"]
            strategy._pending_exit_price = saved["pending_exit_price"]
        elif saved["kind"] == "orb":
            strategy._context = saved["context"]
            for key, value in saved["builder"].items():
                setattr(strategy._context_builder, f"_{key}", value)
            strategy._trade_state = saved["trade_state"]
            strategy._pending_or_high = saved["pending_or_high"]
            strategy._pending_or_low = saved["pending_or_low"]
            strategy._pending_exit_price = saved["pending_exit_price"]
            strategy._in_trade = bool(saved["in_trade"])
        else:
            strategy._in_trade = bool(saved["in_trade"])
            strategy._entry_price = saved["entry_price"]
            strategy._entry_bar = saved["entry_bar"]
            strategy._last_bar_index = int(saved["last_bar_index"])
            strategy._pending_exit_price = saved["pending_exit_price"]
            recovery = strategy._recovery
            for key, value in saved["recovery"].items():
                setattr(recovery, f"_{key}" if key == "last_bar" else key, value)
            strategy.fitted_model = saved["fitted_model"]
            strategy.model_window = saved["model_window"]


def _validate_pending_execution(engine: Any) -> None:
    """Reject a checkpoint whose unsettled order references cannot be resumed."""
    broker_ids = set(engine.broker._orders)
    coordinator_ids = set(engine.broker_execution._broker_orders)
    strategy_ids = {
        order_id for order_ids in engine._pending_strategy_orders.values()
        for order_id in order_ids
    }
    pending_risk_ids = set(engine._pending_risk_results)
    pending_entry_ids = set(engine._pending_entry_requests)
    simulated_ids = set(engine._simulated_orders)
    if not (coordinator_ids <= broker_ids and strategy_ids <= broker_ids
            and strategy_ids <= coordinator_ids):
        raise ValueError("checkpoint pending-order references are inconsistent")
    if not (pending_risk_ids == pending_entry_ids
            and pending_risk_ids <= strategy_ids
            and simulated_ids <= strategy_ids):
        raise ValueError("checkpoint pending-entry authorization is inconsistent")
    for order_id in pending_risk_ids:
        submission = engine.broker_execution._broker_orders[order_id]
        if submission.execution_intent.action.value != "enter":
            raise ValueError("checkpoint pending-entry authorization references a non-entry order")


class AtomicCheckpointStore:
    """Checksummed JSON checkpoints with atomic replacement and fallback copy."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")

    def save(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "payload": payload,
        }
        body = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=True)
        document = json.dumps({**envelope, "sha256": sha256(body.encode()).hexdigest()}, sort_keys=True, separators=(",", ":"), allow_nan=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        if self.path.exists():
            os.replace(self.path, self.backup_path)
        os.replace(temporary, self.path)
        try:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Windows does not consistently permit fsync on directory handles.
            pass

    def load(self) -> dict[str, Any]:
        errors: list[str] = []
        for candidate in (self.path, self.backup_path):
            if not candidate.exists():
                continue
            try:
                document = json.loads(candidate.read_text(encoding="utf-8"))
                envelope = {
                    "schema_version": document["schema_version"],
                    "payload": document["payload"],
                }
                body = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=True)
                if sha256(body.encode()).hexdigest() != document.get("sha256"):
                    raise ValueError("checkpoint checksum mismatch")
                if int(document["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
                    raise ValueError("unsupported checkpoint schema")
                return dict(document["payload"])
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
        if errors:
            raise ValueError("No valid checkpoint: " + "; ".join(errors))
        raise FileNotFoundError(self.path)
