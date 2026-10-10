"""Progress and reusable feature artifacts for future Paper bootstraps.

This module deliberately does not contain Paper execution checkpoint state.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


FEATURE_CACHE_VERSION = 2
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class BootstrapProgressWriter:
    """Write throttled, atomic progress snapshots outside Paper service state."""

    def __init__(self, path: str | Path, *, flush_interval_seconds: float = 10.0,
                 flush_interval_rows: int = 10_000) -> None:
        if flush_interval_seconds <= 0 or flush_interval_rows <= 0:
            raise ValueError("Progress flush intervals must be positive.")
        self.path = Path(path)
        self.flush_interval_seconds = float(flush_interval_seconds)
        self.flush_interval_rows = int(flush_interval_rows)
        self.started = time.monotonic()
        self._stage_started = self.started
        self._last_flush = 0.0
        self._last_flushed_rows = 0
        self._payload: dict[str, Any] = {
            "schema_version": 1,
            "updated_at_utc": _utc_now(),
            "run_started_at_utc": _utc_now(),
            "current_stage": "starting",
            "stage_started_at_utc": _utc_now(),
            "stage_elapsed_seconds": 0.0,
            "stage_elapsed_seconds_by_stage": {},
            "historical_rows_processed": 0,
            "historical_rows_total": None,
            "last_processed_timestamp_utc": None,
            "current_hmm_refit_status": "not_started",
            "current_refit_timestamp_utc": None,
            "run_elapsed_seconds": 0.0,
            "completed": False,
        }
        self._write(force=True)

    @property
    def current_stage(self) -> str:
        return str(self._payload["current_stage"])

    def annotate(self, key: str, value: Any) -> None:
        self._payload.setdefault("details", {})[key] = value
        self._write(force=True)

    def stage(self, name: str, *, total_rows: int | None = None) -> None:
        now = time.monotonic()
        previous = str(self._payload["current_stage"])
        self._payload["stage_elapsed_seconds_by_stage"][previous] = round(
            now - self._stage_started, 6
        )
        self._payload.update({
            "current_stage": name,
            "stage_elapsed_seconds": 0.0,
            "historical_rows_processed": 0,
            "historical_rows_total": int(total_rows) if total_rows is not None else None,
            "last_processed_timestamp_utc": None,
        })
        self._stage_started = now
        self._payload["stage_started_at_utc"] = _utc_now()
        self._last_flushed_rows = 0
        self._write(force=True)

    def update(self, *, rows_processed: int | None = None,
               total_rows: int | None = None,
               last_timestamp: Any = None,
               hmm_refit_status: str | None = None,
               refit_timestamp: Any = None,
               force: bool = False) -> None:
        if rows_processed is not None:
            self._payload["historical_rows_processed"] = int(rows_processed)
        if total_rows is not None:
            self._payload["historical_rows_total"] = int(total_rows)
        if last_timestamp is not None:
            stamp = pd.Timestamp(last_timestamp)
            if stamp.tzinfo is None:
                raise ValueError("Bootstrap progress timestamps must be timezone-aware.")
            self._payload["last_processed_timestamp_utc"] = stamp.tz_convert("UTC").isoformat()
        if hmm_refit_status is not None:
            self._payload["current_hmm_refit_status"] = hmm_refit_status
        if refit_timestamp is not None:
            refit_stamp = pd.Timestamp(refit_timestamp)
            if refit_stamp.tzinfo is None:
                raise ValueError("HMM refit progress timestamps must be timezone-aware.")
            self._payload["current_refit_timestamp_utc"] = refit_stamp.tz_convert("UTC").isoformat()
        elif hmm_refit_status in {"idle", "not_started"}:
            self._payload["current_refit_timestamp_utc"] = None
        self._write(force=force)

    def finish(self) -> None:
        now = time.monotonic()
        stage = str(self._payload["current_stage"])
        self._payload["stage_elapsed_seconds_by_stage"][stage] = round(
            now - self._stage_started, 6
        )
        self._payload.update({
            "current_stage": "completed",
            "stage_elapsed_seconds": 0.0,
            "run_elapsed_seconds": round(now - self.started, 6),
            "completed": True,
            "current_hmm_refit_status": "idle",
        })
        self._write(force=True)

    def _write(self, *, force: bool) -> None:
        now = time.monotonic()
        elapsed = now - self._last_flush
        rows_delta = int(self._payload["historical_rows_processed"]) - self._last_flushed_rows
        # Stage transitions and refit start/end notifications are forced;
        # regular row updates are throttled to bound disk writes.
        status = str(self._payload["current_hmm_refit_status"])
        refit_transition = status.endswith("in_progress") or status.endswith("completed")
        if not force and not refit_transition and elapsed < self.flush_interval_seconds and rows_delta < self.flush_interval_rows:
            return
        self._payload["updated_at_utc"] = _utc_now()
        self._payload["stage_elapsed_seconds"] = round(now - self._stage_started, 6)
        self._payload["run_elapsed_seconds"] = round(now - self.started, 6)
        _atomic_json(self.path, self._payload)
        self._last_flush = now
        self._last_flushed_rows = int(self._payload["historical_rows_processed"])


def _feature_code_identity() -> str:
    from src.feature_engine import add_return_features, add_volatility_features
    from src.paper.market_context import build_causal_context_features
    from src.research.direction.direction_features import (
        add_directional_pressure_features,
        add_normalized_momentum_features,
        add_range_location_features,
    )

    functions = (
        build_causal_context_features,
        build_replay_causal_features,
        add_return_features,
        add_volatility_features,
        add_directional_pressure_features,
        add_range_location_features,
        add_normalized_momentum_features,
    )
    source = "\n".join(inspect.getsource(function) for function in functions)
    from src.paper.market_context import PRECOMPUTED_CONTEXT_FEATURES
    try:
        tzdata_version = package_metadata.version("tzdata")
    except package_metadata.PackageNotFoundError:
        tzdata_version = "platform-zoneinfo"
    identity = {
        "cache_version": FEATURE_CACHE_VERSION,
        "code_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "tzdata": tzdata_version,
        "feature_columns": PRECOMPUTED_CONTEXT_FEATURES,
        "timezone": "America/New_York",
        "directional_warmup_returns": 30,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=list).encode()).hexdigest()


def _source_data_identity(raw: pd.DataFrame, *, chunk_rows: int = 100_000,
                          progress: Callable[[int, int, Any], None] | None = None) -> str:
    timestamp_column = "timestamp ET" if "timestamp ET" in raw else "timestamp" if "timestamp" in raw else None
    required = {"open", "high", "low", "close", "volume"}
    if timestamp_column is None or not required.issubset(raw.columns):
        raise ValueError("Feature cache requires canonical timestamp and OHLCV columns.")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")
    count = len(raw)
    digest = hashlib.sha256()
    identity_columns = [timestamp_column, *sorted(required)]
    # Instrument/source identity does not change the numerical feature formula,
    # but it must change the cache key: the same OHLCV values from a different
    # contract or provider are not interchangeable bootstrap inputs.
    source_identity_columns = [
        name for name in ("symbol", "instrument_id", "contract_id", "dataset", "schema")
        if name in raw.columns
    ]
    digest.update(json.dumps({
        "rows": count,
        "columns": [*identity_columns, *source_identity_columns],
    }, sort_keys=True).encode())
    if "market_period" in raw:
        digest.update(b"market_period\0")
    previous: pd.Timestamp | None = None
    for start in range(0, count, chunk_rows):
        block = raw.iloc[start:start + chunk_rows]
        stamps = pd.to_datetime(block[timestamp_column], utc=True, errors="raise")
        if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
            raise ValueError("Feature cache input timestamps must be non-null, unique, and ordered.")
        if previous is not None and stamps.size and stamps.iloc[0] <= previous:
            raise ValueError("Feature cache input timestamps must be globally increasing.")
        if stamps.size:
            previous = stamps.iloc[-1]
        normalized: dict[str, Any] = {"timestamp_ns": stamps.astype("int64").to_numpy()}
        for name in ("open", "high", "low", "close", "volume"):
            values = pd.to_numeric(block[name], errors="raise").to_numpy(dtype="float64")
            if not np.isfinite(values).all():
                raise ValueError(f"Feature cache input {name} contains non-finite values.")
            normalized[name] = values
        if "market_period" in block:
            normalized["market_period"] = block["market_period"].astype("string").fillna("").to_numpy()
        for name in source_identity_columns:
            values = block[name]
            if name == "instrument_id":
                numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype="float64")
                if not np.isfinite(numeric).all():
                    raise ValueError("Feature cache instrument_id contains non-finite values.")
                normalized[name] = numeric
            else:
                normalized[name] = values.astype("string").fillna("").to_numpy()
        row_hashes = pd.util.hash_pandas_object(pd.DataFrame(normalized), index=False).to_numpy(dtype="<u8")
        digest.update(row_hashes.tobytes())
        if progress is not None:
            progress(min(start + len(block), count), count, stamps.iloc[-1] if len(stamps) else None)
    return digest.hexdigest()


def build_replay_causal_features(
    raw: pd.DataFrame,
    *,
    progress_callback: Callable[[str, int, int, Any], None] | None = None,
) -> pd.DataFrame:
    """Run the existing causal feature builder and its replay warmup guard."""
    from src.paper.market_context import build_causal_context_features

    features = build_causal_context_features(raw, progress_callback=progress_callback)
    directional_ready = features["log_return"].rolling(30).count() == 30
    features.loc[~directional_ready, "close_location_30"] = float("nan")
    return features


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_or_build_causal_feature_cache(
    raw: pd.DataFrame,
    cache_dir: str | Path,
    *,
    progress: Callable[[str, int, int, Any], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a verified Parquet feature cache or rebuild it exactly.

    Cache identity covers ordered input values, feature implementation,
    feature schema, timezone rules, and pandas/NumPy versions. Invalid or
    partial cache files are rejected and atomically replaced after recompute.
    """
    root = Path(cache_dir)
    if not len(raw):
        raise ValueError("Cannot cache features for an empty historical interval.")

    def source_progress(rows: int, total: int, stamp: Any) -> None:
        if progress:
            progress("feature_cache_identity", rows, total, stamp)

    source_hash = _source_data_identity(raw, progress=source_progress)
    code_hash = _feature_code_identity()
    identity = {
        "cache_schema": FEATURE_CACHE_VERSION,
        "source_data_sha256": source_hash,
        "feature_code_identity": code_hash,
        "rows": len(raw),
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    parquet_path = root / f"causal_features_{cache_key}.parquet"
    manifest_path = root / f"causal_features_{cache_key}.json"

    if progress:
        progress("feature_cache_lookup", 0, len(raw), None)
    rejected_cache_reason: str | None = None
    if manifest_path.exists() or parquet_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity") != identity:
                raise ValueError("feature cache manifest identity mismatch")
            if not parquet_path.is_file() or _file_sha256(parquet_path) != manifest.get("parquet_sha256"):
                raise ValueError("feature cache data is missing or checksum-invalid")
            frame = pd.read_parquet(parquet_path, engine="pyarrow")
            columns = manifest.get("columns")
            if (len(frame) != len(raw) or not frame.columns.is_unique
                    or list(frame.columns) != columns
                    or {name: str(dtype) for name, dtype in frame.dtypes.items()} != manifest.get("dtypes")
                    or not {"timestamp", "timestamp ET"}.issubset(frame.columns)):
                raise ValueError("feature cache row count/schema is incompatible")
            stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
            expected_col = "timestamp ET" if "timestamp ET" in raw else "timestamp"
            expected = pd.to_datetime(raw[expected_col], utc=True, errors="raise").reset_index(drop=True)
            if (stamps.isna().any() or stamps.duplicated().any()
                    or not stamps.is_monotonic_increasing
                    or not stamps.reset_index(drop=True).equals(expected)):
                raise ValueError("feature cache timestamps are incompatible")
            if progress:
                progress("feature_cache_loaded", len(raw), len(raw), stamps.iloc[-1])
            return frame, {"status": "hit", "cache_key": cache_key, "rows": len(frame), "path": str(parquet_path)}
        except Exception as exc:
            rejected_cache_reason = f"{type(exc).__name__}: {exc}"

    if progress:
        progress("feature_construction", 0, len(raw), None)
    def feature_progress(stage: str, rows: int, total: int, stamp: Any) -> None:
        if progress:
            progress(stage, rows, total, stamp)

    frame = build_replay_causal_features(raw, progress_callback=feature_progress)
    if len(frame) != len(raw) or not frame.columns.is_unique:
        raise ValueError("Feature builder changed historical row count or produced duplicate columns.")
    expected_col = "timestamp ET" if "timestamp ET" in raw else "timestamp"
    expected = pd.to_datetime(raw[expected_col], utc=True, errors="raise").reset_index(drop=True)
    stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise").reset_index(drop=True)
    if not stamps.equals(expected):
        raise ValueError("Feature builder changed historical timestamp order or coverage.")
    try:
        _atomic_parquet(frame, parquet_path)
        manifest = {
            "identity": identity,
            "cache_key": cache_key,
            "rows": len(frame),
            "columns": list(frame.columns),
            "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
            "first_timestamp_utc": stamps.iloc[0].isoformat(),
            "last_timestamp_utc": stamps.iloc[-1].isoformat(),
            "parquet_sha256": _file_sha256(parquet_path),
            "created_at_utc": _utc_now(),
        }
        _atomic_json(manifest_path, manifest)
    except Exception as exc:
        if progress:
            progress("feature_cache_write_failed_using_memory", len(frame), len(frame), stamps.iloc[-1])
        return frame, {
            "status": "miss_built_cache_write_failed",
            "cache_key": cache_key,
            "rows": len(frame),
            "path": str(parquet_path),
            "write_error": f"{type(exc).__name__}: {exc}",
        }
    if progress:
        progress("feature_cache_written", len(frame), len(frame), stamps.iloc[-1])
    return frame, {
        "status": "miss_rebuilt" if rejected_cache_reason else "miss_built",
        "rejected_cache_reason": rejected_cache_reason,
        "cache_key": cache_key,
        "rows": len(frame),
        "path": str(parquet_path),
    }


__all__ = [
    "BootstrapProgressWriter",
    "build_replay_causal_features",
    "load_or_build_causal_feature_cache",
]
