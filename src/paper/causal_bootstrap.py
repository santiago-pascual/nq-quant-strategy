"""Versioned, resumable causal context warm-up for a fresh PAPER account.

This artifact contains market context/HMM state only. It intentionally does
not contain Paper engine, strategy lifecycle, order, position, risk, or PnL
state. A new account must be constructed independently at activation.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.paper.bootstrap_artifacts import _feature_code_identity
from src.paper.market_context import CausalMarketContext
from src.paper.realtime_checkpoint import runtime_identity


BOOTSTRAP_SCHEMA_VERSION = 2
_REQUIRED_FEATURES = (
    "timestamp", "timestamp ET", "open", "high", "low", "close", "volume",
    "log_return", "realized_vol_5", "realized_vol_15", "realized_vol_30",
    "realized_vol_60", "variance_ratio_5_30", "variance_ratio_5_60",
    "past_return_30", "directional_pressure_30", "close_location_30",
    "normalized_momentum_30",
)


def _aware_utc(value: Any, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _json_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False, default=str)
    return sha256(body.encode("utf-8")).hexdigest()


def _hash_frame(frame: pd.DataFrame, *, chunk_rows: int = 50_000) -> str:
    if not frame.columns.is_unique:
        raise ValueError("Bootstrap source has duplicate column names")
    digest = sha256()
    digest.update(_json_hash({
        "columns": list(frame.columns),
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "rows": len(frame),
    }).encode("ascii"))
    for start in range(0, len(frame), chunk_rows):
        block = frame.iloc[start:start + chunk_rows].reset_index(drop=True)
        digest.update(pd.util.hash_pandas_object(
            block, index=False, categorize=True
        ).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def _context_schedule_identity(context: CausalMarketContext) -> dict[str, Any]:
    provider = context._raw_hmm_provider
    streams = {}
    for name, stream in (("MR", provider.mr), ("S2R", provider.s2r)):
        streams[name] = {
            "schedule": stream.schedule,
            "refit_months": stream.refit_months,
            "rolling_years": stream.rolling_years,
            "eligible_rth_only": stream.eligible_rth_only,
            "model_config": asdict(stream.config),
        }
    return {"market_context_config": asdict(context.config), "hmm_streams": streams}


def _fresh_strategy_state_seed() -> dict[str, Any]:
    """Describe the real constructors' fresh-account state, not warm-up trades.

    The bootstrap advances only causal context/HMM. It must not replay strategy
    executions or inherit historical positions. These explicit defaults are
    the state contract consumed by a newly constructed Paper engine at
    activation.
    """
    from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
    from src.strategies.mean_reversion.strategy import MeanReversionStrategy
    from src.strategies.orb.strategy import ORBStrategy
    from src.strategies.s2r.strategy import S2RStrategy

    instances = (
        MeanReversionStrategy(MRL1_CONFIG),
        MeanReversionStrategy(MRS2_CONFIG),
        S2RStrategy(),
        ORBStrategy(),
    )
    names = [strategy.name for strategy in instances]
    if names != ["MRL1", "MRS2", "S2R", "ORB"]:
        raise RuntimeError(f"Unexpected fresh Paper strategy set: {names}")
    states = {}
    for strategy in instances:
        if isinstance(strategy, MeanReversionStrategy):
            state = {
                "lifecycle": "flat",
                "trade_state": None,
                "pending_exit_price": None,
            }
            if strategy._trade_state is not None or strategy._pending_exit_price is not None:
                raise RuntimeError(f"{strategy.name} constructor did not create a fresh state")
        elif isinstance(strategy, S2RStrategy):
            state = {
                "lifecycle": "flat",
                "in_trade": bool(strategy._in_trade),
                "entry_price": strategy._entry_price,
                "entry_bar": strategy._entry_bar,
                "pending_exit_price": strategy._pending_exit_price,
                "recovery_tracker": {
                    "state": strategy._recovery.state.value,
                    "mae_bar": strategy._recovery.mae_bar,
                    "recovery_bar": strategy._recovery.recovery_bar,
                    "exit_bar": strategy._recovery.exit_bar,
                    "last_bar": strategy._recovery._last_bar,
                },
            }
            if strategy._in_trade or strategy._entry_price is not None or strategy._entry_bar is not None:
                raise RuntimeError("S2R constructor did not create a fresh state")
        else:
            state = {
                "lifecycle": "flat",
                "in_trade": bool(strategy._in_trade),
                "trade_state": None,
                "orb_context": None,
                "pending_or_high": strategy._pending_or_high,
                "pending_or_low": strategy._pending_or_low,
                "pending_exit_price": strategy._pending_exit_price,
            }
            if strategy._in_trade or strategy._trade_state is not None or strategy._context is not None:
                raise RuntimeError("ORB constructor did not create a fresh state")
        states[strategy.name] = {
            "strategy_version": strategy.version,
            "strategy_configuration": asdict(strategy.config),
            "state": state,
        }
    source_files = (
        Path(__file__).resolve().parents[1] / "strategies/mean_reversion/strategy.py",
        Path(__file__).resolve().parents[1] / "strategies/mean_reversion/config.py",
        Path(__file__).resolve().parents[1] / "strategies/s2r/strategy.py",
        Path(__file__).resolve().parents[1] / "strategies/s2r/config.py",
        Path(__file__).resolve().parents[1] / "strategies/orb/strategy.py",
        Path(__file__).resolve().parents[1] / "strategies/orb/config.py",
    )
    source_hashes = {
        str(path.relative_to(Path(__file__).resolve().parents[1])):
            sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    seed = {
        "schema_version": 1,
        "initialization": "fresh_strategy_constructors_at_activation",
        "strategies": states,
        "strategy_source_sha256": source_hashes,
        "no_historical_strategy_execution": True,
    }
    # Normalize tuples/enums and other JSON-compatible configuration values so
    # checkpoint identity compares identically after a process restart.
    return json.loads(json.dumps(seed, sort_keys=True, default=str, allow_nan=False))


def _numerical_thread_identity() -> list[dict[str, Any]]:
    """Require deterministic single-thread BLAS before any HMM fit occurs."""
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    identity = [
        {
            "user_api": str(pool.get("user_api", "")),
            "internal_api": str(pool.get("internal_api", "")),
            "num_threads": int(pool.get("num_threads", 0)),
            "version": str(pool.get("version", "")),
        }
        for pool in pools
    ]
    offenders = [pool for pool in identity if pool["num_threads"] != 1]
    if offenders:
        raise RuntimeError(
            "Causal bootstrap requires single-thread numerical pools for repeatable HMM fits; "
            f"found {offenders}. Set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, "
            "and MKL_NUM_THREADS=1 before starting Python."
        )
    return identity


class CausalBootstrapCheckpointStore:
    """Atomic and checksummed bootstrap checkpoint, separate from Paper state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, payload: Mapping[str, Any]) -> None:
        envelope = {"schema_version": BOOTSTRAP_SCHEMA_VERSION,
                    "payload": dict(payload)}
        body = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                          allow_nan=True, default=str)
        document = {**envelope, "sha256": sha256(body.encode("utf-8")).hexdigest()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(document, stream, sort_keys=True, separators=(",", ":"),
                          allow_nan=True, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load(self, *, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
        document = json.loads(self.path.read_text(encoding="utf-8"))
        envelope = {"schema_version": document.get("schema_version"),
                    "payload": document.get("payload")}
        body = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                          allow_nan=True, default=str)
        if document.get("sha256") != sha256(body.encode("utf-8")).hexdigest():
            raise ValueError("causal bootstrap checkpoint checksum mismatch")
        if int(document.get("schema_version", 0)) != BOOTSTRAP_SCHEMA_VERSION:
            raise ValueError("unsupported causal bootstrap checkpoint schema")
        payload = dict(document["payload"])
        if payload.get("identity") != dict(expected_identity):
            raise ValueError("causal bootstrap checkpoint identity differs from source/config")
        rows = int(payload.get("rows_processed", -1))
        if rows < 0:
            raise ValueError("causal bootstrap checkpoint has invalid row cursor")
        last = payload.get("last_processed_timestamp_utc")
        context = payload.get("context")
        if rows and (last is None or not isinstance(context, Mapping)):
            raise ValueError("causal bootstrap checkpoint is incomplete")
        return payload


class CausalBootstrapRunner:
    """Advance only causal features/HMMs and checkpoint at bounded row batches."""

    def __init__(
        self,
        *,
        context: CausalMarketContext,
        checkpoint_path: str | Path,
        activation_timestamp: Any,
        history_origin_timestamp: Any,
        feature_cache_key: str,
        initial_equity: float,
        risk_configuration: Mapping[str, Any],
        coverage_validator: Callable[[pd.DataFrame], Mapping[str, Any]],
        chunk_rows: int = 10_000,
    ) -> None:
        self.context = context
        self.store = CausalBootstrapCheckpointStore(checkpoint_path)
        self.activation = _aware_utc(activation_timestamp, "activation timestamp")
        self.origin = _aware_utc(history_origin_timestamp, "history origin timestamp")
        if not feature_cache_key:
            raise ValueError("A verified causal feature-cache key is required")
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            raise ValueError("New Paper account initial equity must be positive and finite")
        if not risk_configuration:
            raise ValueError("Original Paper risk configuration must be explicit")
        if chunk_rows <= 0:
            raise ValueError("Bootstrap chunk_rows must be positive")
        if not callable(coverage_validator):
            raise ValueError("A verified historical coverage validator is required")
        self.feature_cache_key = str(feature_cache_key)
        self.initial_equity = float(initial_equity)
        self.risk_configuration = dict(risk_configuration)
        self.coverage_validator = coverage_validator
        self.chunk_rows = int(chunk_rows)

    def _validate_inputs(
        self, raw: pd.DataFrame, features: pd.DataFrame
    ) -> tuple[pd.Series, dict[str, Any], dict[str, Any]]:
        timestamp_col = "timestamp" if "timestamp" in raw else "timestamp ET" if "timestamp ET" in raw else None
        if timestamp_col is None or raw.empty:
            raise ValueError("Bootstrap requires nonempty raw history with timestamps")
        if not raw.columns.is_unique or not features.columns.is_unique:
            raise ValueError("Bootstrap inputs have duplicate column names")
        stamps = pd.to_datetime(raw[timestamp_col], utc=True, errors="raise")
        if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
            raise ValueError("Bootstrap raw timestamps must be unique and strictly ordered")
        if stamps.iloc[0] != self.origin:
            raise ValueError("Bootstrap history does not begin at the configured historical origin")
        if stamps.iloc[-1] >= self.activation:
            raise ValueError("Bootstrap source includes activation-time or future observations")
        if not set(_REQUIRED_FEATURES).issubset(features.columns):
            raise ValueError("Bootstrap feature frame does not contain the complete causal schema")
        feature_stamps = pd.to_datetime(features["timestamp"], utc=True, errors="raise")
        if (len(features) != len(raw) or feature_stamps.isna().any()
                or feature_stamps.duplicated().any() or not feature_stamps.is_monotonic_increasing
                or not feature_stamps.reset_index(drop=True).equals(stamps.reset_index(drop=True))):
            raise ValueError("Bootstrap features do not exactly match raw ordered timestamps")
        coverage = dict(self.coverage_validator(raw))
        strict_coverage_verified = (
            coverage.get("verified") is True
            and coverage.get("missing_expected_minutes", 0) == 0
            and coverage.get("unknown_coverage", 0) == 0
            and coverage.get("certificate_readiness") == "FULLY_CERTIFIED"
        )
        paper_acceptance = coverage.get("paper_coverage_acceptance", {})
        accepted_paper_coverage = (
            paper_acceptance.get("policy") == "paper_research_quality_accepted"
            and paper_acceptance.get("decision") == "ACCEPTED_FOR_INTERNAL_SIMULATED_PAPER_ONLY"
            and paper_acceptance.get("scope") == "internal_simulated_paper_only_no_live_execution"
            and bool(paper_acceptance.get("acceptance_artifact_sha256"))
            and bool(paper_acceptance.get("strict_certificate_sha256"))
            and paper_acceptance.get("observed_data_integrity_verified") is True
            and paper_acceptance.get("demonstrated_missing_required_observations") == 0
            and paper_acceptance.get("unresolved_span_count") == 8
            and paper_acceptance.get("unresolved_absent_minutes") == 1051
            and coverage.get("certificate_readiness") == "NOT_FULLY_CERTIFIED"
            and coverage.get("verified") is True
            and coverage.get("missing_expected_minutes", 0) == 0
            and coverage.get("unknown_coverage", 0) == 0
        )
        if not (strict_coverage_verified or accepted_paper_coverage):
            raise ValueError("Historical session coverage is not fully verified")
        if self.context._bars_seen or self.context.last_timestamp is not None:
            raise RuntimeError("Bootstrap runner requires a fresh context; use resume=True for restore")
        source_identity = {
            "source_frame_sha256": _hash_frame(raw),
            "rows": len(raw),
            "first_timestamp_utc": stamps.iloc[0].isoformat(),
            "last_timestamp_utc": stamps.iloc[-1].isoformat(),
            "columns": list(raw.columns),
            "coverage_certificate_sha256": _json_hash(coverage),
        }
        config_identity = {
            "causal_feature_code_identity": _feature_code_identity(),
            "runtime_identity": runtime_identity(),
            "numerical_thread_pools": _numerical_thread_identity(),
            "bootstrap_driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "feature_cache_key": self.feature_cache_key,
            "feature_frame_sha256": _hash_frame(features),
            "context_and_schedule": _context_schedule_identity(self.context),
            "fresh_strategy_state_seed": _fresh_strategy_state_seed(),
            "activation_timestamp_utc": self.activation.isoformat(),
            "history_origin_timestamp_utc": self.origin.isoformat(),
            "new_account": {
                "initial_equity": self.initial_equity,
                "risk_configuration": self.risk_configuration,
                "strategy_state_included": True,
                "positions_included": False,
                "orders_included": False,
                "trades_included": False,
            },
        }
        identity = {"source": source_identity, "configuration": config_identity}
        return stamps, identity, coverage

    def run(
        self,
        raw: pd.DataFrame,
        features: pd.DataFrame,
        *,
        resume: bool = False,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        stamps, identity, coverage = self._validate_inputs(raw, features)
        if resume:
            payload = self.store.load(expected_identity=identity)
            rows_done = int(payload["rows_processed"])
            expected_last = stamps.iloc[rows_done - 1].isoformat() if rows_done else None
            if payload.get("last_processed_timestamp_utc") != expected_last:
                raise ValueError("Bootstrap checkpoint cursor does not match source rows")
            if rows_done and self.context._bars_seen == 0:
                self.context.load_state_dict(payload["context"])
            elif rows_done and self.context.last_timestamp != pd.Timestamp(expected_last):
                raise ValueError("Restored causal context differs from bootstrap cursor")
            if not rows_done and self.context._bars_seen:
                raise ValueError("Empty bootstrap cursor cannot restore a populated context")
        else:
            if self.store.path.exists():
                raise FileExistsError("Bootstrap checkpoint exists; explicitly resume it or use a new path")
            rows_done = 0

        while rows_done < len(raw):
            stop = min(rows_done + self.chunk_rows, len(raw))
            self.context.bootstrap_causal_history(
                raw.iloc[rows_done:stop].copy(),
                precomputed_features=features.iloc[rows_done:stop].copy(),
                resume=rows_done > 0,
                total_rows=len(raw),
                progress_callback=(
                    lambda event: progress_callback(event)
                    if progress_callback is not None else None
                ),
            )
            rows_done = stop
            last = stamps.iloc[rows_done - 1].isoformat()
            payload = {
                "artifact_schema_version": BOOTSTRAP_SCHEMA_VERSION,
                "identity": identity,
                "rows_processed": rows_done,
                "last_processed_timestamp_utc": last,
                "activation_timestamp_utc": self.activation.isoformat(),
                "history_origin_timestamp_utc": self.origin.isoformat(),
                "coverage_certificate": coverage,
                "account_seed": {
                    "initial_equity": self.initial_equity,
                    "risk_configuration": self.risk_configuration,
                    "positions": [], "orders": [], "trades": [],
                },
                "strategy_state_seed": identity["configuration"]["fresh_strategy_state_seed"],
                "context": self.context.state_dict(),
                "ready_for_activation": False,
                "written_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.store.save(payload)
            if progress_callback is not None:
                progress_callback({"rows_processed": rows_done, "total_rows": len(raw),
                                   "last_timestamp": stamps.iloc[rows_done - 1],
                                   "checkpoint_committed": True})

        if self.context.last_timestamp is None or self.context.last_timestamp >= self.activation:
            raise RuntimeError("Causal warm-up did not stop strictly before activation")
        final = self.store.load(expected_identity=identity)
        final["ready_for_activation"] = True
        final["activation_checkpoint_written_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.store.save(final)
        return final


def load_context_seed_for_new_account(
    *,
    store: CausalBootstrapCheckpointStore,
    context: CausalMarketContext,
    expected_identity: Mapping[str, Any],
    activation_timestamp: Any,
) -> dict[str, Any]:
    """Restore warmed market context into a fresh account, never engine state."""
    activation = _aware_utc(activation_timestamp, "activation timestamp")
    if context._bars_seen or context.last_timestamp is not None:
        raise RuntimeError("New Paper account context must be empty before seed restore")
    payload = store.load(expected_identity=expected_identity)
    errors = activation_payload_errors(payload, activation_timestamp=activation,
                                       expected_identity=expected_identity)
    if errors:
        raise ValueError("Causal bootstrap artifact failed activation schema validation: "
                         + "; ".join(errors))
    if _aware_utc(payload["activation_timestamp_utc"], "artifact activation") != activation:
        raise ValueError("Activation timestamp differs from warm-up artifact")
    context.load_state_dict(payload["context"])
    if context.last_timestamp is None or context.last_timestamp >= activation:
        raise ValueError("Warm context is not strictly prior to account activation")
    return payload["account_seed"]


def activation_payload_errors(
    payload: Mapping[str, Any], *, activation_timestamp: Any,
    expected_identity: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the versioned delayed-Paper activation seed contract."""
    errors: list[str] = []
    if int(payload.get("artifact_schema_version", 0)) != BOOTSTRAP_SCHEMA_VERSION:
        errors.append("unsupported activation artifact schema")
    if payload.get("ready_for_activation") is not True:
        errors.append("artifact is not marked ready_for_activation")
    if expected_identity is not None and payload.get("identity") != dict(expected_identity):
        errors.append("trusted identity manifest differs from artifact identity")
    identity = payload.get("identity", {})
    if not identity.get("source", {}).get("source_frame_sha256"):
        errors.append("source-data fingerprint is missing")
    configuration = identity.get("configuration", {})
    if not configuration.get("causal_feature_code_identity") or not configuration.get("context_and_schedule"):
        errors.append("feature/HMM configuration fingerprint is missing")
    if not configuration.get("runtime_identity"):
        errors.append("runtime fingerprint is missing")
    if not identity.get("source", {}).get("coverage_certificate_sha256"):
        errors.append("coverage-certificate fingerprint is missing")
    coverage = payload.get("coverage_certificate", {})
    calendar_document = coverage.get("calendar_coverage_certificate", {})
    strict_coverage = (
        coverage.get("verified") is True
        and coverage.get("missing_expected_minutes", 0) == 0
        and coverage.get("unknown_coverage", 0) == 0
        and coverage.get("certificate_readiness") == "FULLY_CERTIFIED"
        and calendar_document.get("readiness") == "FULLY_CERTIFIED"
        and bool(calendar_document.get("calendar_snapshots"))
    )
    paper_acceptance = coverage.get("paper_coverage_acceptance", {})
    accepted_paper_coverage = (
        paper_acceptance.get("policy") == "paper_research_quality_accepted"
        and paper_acceptance.get("decision") == "ACCEPTED_FOR_INTERNAL_SIMULATED_PAPER_ONLY"
        and paper_acceptance.get("scope") == "internal_simulated_paper_only_no_live_execution"
        and paper_acceptance.get("acceptance_artifact_sha256")
        and paper_acceptance.get("strict_certificate_sha256")
        and paper_acceptance.get("observed_data_integrity_verified") is True
        and paper_acceptance.get("demonstrated_missing_required_observations") == 0
        and paper_acceptance.get("unresolved_span_count") == 8
        and paper_acceptance.get("unresolved_absent_minutes") == 1051
        and coverage.get("verified") is True
        and coverage.get("missing_expected_minutes", 0) == 0
        and coverage.get("unknown_coverage", 0) == 0
        and coverage.get("certificate_readiness") == "NOT_FULLY_CERTIFIED"
        and calendar_document.get("readiness") == "NOT_FULLY_CERTIFIED"
        and bool(calendar_document.get("calendar_snapshots"))
    )
    if not (strict_coverage or accepted_paper_coverage):
        errors.append("historical coverage certificate is not fully verified for activation")
    elif accepted_paper_coverage:
        try:
            from src.paper.paper_coverage_acceptance import validate_acceptance_record
            repo_root = Path(__file__).resolve().parents[2]
            acceptance_path = repo_root / paper_acceptance["acceptance_artifact_path"]
            certificate_path = repo_root / paper_acceptance["strict_certificate_path"]
            verified_acceptance = validate_acceptance_record(acceptance_path, certificate_path)
            if (verified_acceptance.get("acceptance_artifact_sha256")
                    != paper_acceptance.get("acceptance_artifact_sha256")
                    or verified_acceptance.get("strict_certificate_sha256")
                    != paper_acceptance.get("strict_certificate_sha256")):
                errors.append("Paper-only coverage acceptance identity differs from bootstrap")
        except Exception as exc:
            errors.append(f"Paper-only coverage acceptance cannot be revalidated: {type(exc).__name__}: {exc}")
    context = payload.get("context", {})
    if not context.get("hmm_provider") or int(context.get("version", 0)) < 1:
        errors.append("causal feature/context/HMM state is missing")
    else:
        provider = context["hmm_provider"]
        for name in ("mr", "s2r"):
            if int(provider.get(name, {}).get("fit_count", 0)) < 1:
                errors.append(f"{name.upper()} causal HMM has no fitted model")
    activation = _aware_utc(activation_timestamp, "activation timestamp")
    try:
        last = _aware_utc(payload["last_processed_timestamp_utc"], "last processed timestamp")
        if last >= activation:
            errors.append("last causal observation is not strictly before activation")
    except (KeyError, ValueError, TypeError):
        errors.append("last causal observation timestamp is missing or invalid")
    account = payload.get("account_seed", {})
    try:
        initial_equity = float(account.get("initial_equity", float("nan")))
    except (TypeError, ValueError):
        initial_equity = float("nan")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        errors.append("fresh initial account equity is invalid")
    if not account.get("risk_configuration"):
        errors.append("risk configuration is missing")
    if any(account.get(key) != [] for key in ("positions", "orders", "trades")):
        errors.append("fresh account must not inherit positions, orders, or trades")
    seed = payload.get("strategy_state_seed", {})
    if (seed.get("schema_version") != 1
            or seed.get("initialization") != "fresh_strategy_constructors_at_activation"
            or set(seed.get("strategies", {})) != {"MRL1", "MRS2", "S2R", "ORB"}
            or seed.get("no_historical_strategy_execution") is not True):
        errors.append("complete fresh state for all four strategies is missing")
    return errors


__all__ = [
    "CausalBootstrapCheckpointStore",
    "CausalBootstrapRunner",
    "activation_payload_errors",
    "load_context_seed_for_new_account",
]
