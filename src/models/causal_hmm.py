"""Causal raw-state HMM fitting, filtering, schedules, and checkpoints.

This module is deliberately separate from :mod:`src.models.regime`, whose
scaling and Viterbi behavior are part of the frozen historical Research path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
from hashlib import sha256
import json
import time
import zlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
import scipy
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import StandardScaler

from .regime import HMM_FEATURES


NY = "America/New_York"
FEATURE_SCHEMA = tuple(HMM_FEATURES)
S2R_FEATURES = (
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
    "realized_vol_30",
)


class _CompactObservationHistory:
    """Chronological observations in typed contiguous arrays, not row objects."""

    def __init__(self, columns: tuple[str, ...], *, initial_capacity: int = 4096) -> None:
        self.columns = tuple(columns)
        self.capacity = max(1, int(initial_capacity))
        self.size = 0
        self.timestamps_ns = np.empty(self.capacity, dtype=np.int64)
        self.values = np.empty((self.capacity, len(self.columns)), dtype=np.float64)

    def append(self, timestamp_ns: int, row: Mapping[str, Any]) -> None:
        if self.size == self.capacity:
            self._grow()
        self.timestamps_ns[self.size] = int(timestamp_ns)
        self.values[self.size] = [row.get(name, np.nan) for name in self.columns]
        self.size += 1

    def _grow(self) -> None:
        capacity = self.capacity * 2
        timestamps = np.empty(capacity, dtype=np.int64)
        values = np.empty((capacity, len(self.columns)), dtype=np.float64)
        timestamps[: self.size] = self.timestamps_ns[: self.size]
        values[: self.size] = self.values[: self.size]
        self.timestamps_ns = timestamps
        self.values = values
        self.capacity = capacity

    def reserve(self, rows: int) -> None:
        """Preallocate a known bootstrap length without geometric headroom."""
        required = max(1, int(rows))
        if required == self.capacity:
            return
        if required < self.capacity and self.size:
            return
        timestamps = np.empty(required, dtype=np.int64)
        values = np.empty((required, len(self.columns)), dtype=np.float64)
        timestamps[: self.size] = self.timestamps_ns[: self.size]
        values[: self.size] = self.values[: self.size]
        self.timestamps_ns = timestamps
        self.values = values
        self.capacity = required

    def replace_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        materialized = list(records)
        replacement = _CompactObservationHistory(
            self.columns, initial_capacity=max(1, len(materialized))
        )
        for record in materialized:
            replacement.append(_utc(record["timestamp"]).value, record)
        self.timestamps_ns = replacement.timestamps_ns
        self.values = replacement.values
        self.capacity = replacement.capacity
        self.size = replacement.size

    def retain_from(self, timestamp_ns: int) -> None:
        first = int(np.searchsorted(self.timestamps_ns[: self.size], timestamp_ns, side="left"))
        if first:
            remaining = self.size - first
            self.timestamps_ns[:remaining] = self.timestamps_ns[first:self.size]
            self.values[:remaining] = self.values[first:self.size]
            self.size = remaining

    def frame(self, start: int, stop: int) -> pd.DataFrame:
        timestamps = pd.to_datetime(self.timestamps_ns[start:stop], utc=True)
        data: dict[str, Any] = {
            "timestamp": timestamps,
            "eligible": np.ones(stop - start, dtype=bool),
        }
        for column_index, name in enumerate(self.columns):
            data[name] = self.values[start:stop, column_index]
        return pd.DataFrame(data, copy=False)

    def payload(self) -> dict[str, Any]:
        timestamps = np.ascontiguousarray(self.timestamps_ns[: self.size], dtype="<i8")
        values = np.ascontiguousarray(self.values[: self.size], dtype="<f8")
        packed = zlib.compress(timestamps.tobytes() + values.tobytes(), level=1)
        return {
            "columns": list(self.columns),
            "rows": self.size,
            "timestamp_dtype": "<i8",
            "feature_dtype": "<f8",
            "encoding": "zlib+base64",
            "data": base64.b64encode(packed).decode("ascii"),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "_CompactObservationHistory":
        columns = tuple(payload["columns"])
        rows = int(payload["rows"])
        raw = zlib.decompress(base64.b64decode(payload["data"]))
        timestamp_bytes = rows * np.dtype("<i8").itemsize
        timestamps = np.frombuffer(raw[:timestamp_bytes], dtype="<i8").copy()
        values = np.frombuffer(raw[timestamp_bytes:], dtype="<f8").reshape(
            rows, len(columns)
        ).copy()
        result = cls(columns, initial_capacity=max(1, rows))
        result.timestamps_ns[:rows] = timestamps
        result.values[:rows] = values
        result.size = rows
        return result

    def records(self) -> list[dict[str, Any]]:
        timestamps = pd.to_datetime(self.timestamps_ns[: self.size], utc=True)
        return [
            {
                "timestamp": timestamp,
                **dict(zip(self.columns, row)),
                "eligible": True,
            }
            for timestamp, row in zip(timestamps, self.values[: self.size])
        ]

    def fingerprint(self, start: int, stop: int) -> str:
        digest = sha256()
        digest.update(json.dumps(self.columns).encode())
        if start == stop:
            return digest.hexdigest()
        digest.update(memoryview(self.timestamps_ns[start:stop]).cast("B"))
        digest.update(memoryview(self.values[start:stop]).cast("B"))
        return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Causal HMM timestamps must be timezone-aware.")
    return stamp.tz_convert("UTC")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def add_calendar_months(timestamp: Any, months: int) -> pd.Timestamp:
    """Add calendar months in New York wall time, returning a UTC timestamp."""
    local = _utc(timestamp).tz_convert(NY)
    naive = local.tz_localize(None) + pd.DateOffset(months=int(months))
    # Fit deadlines are normally at observed bar times. Resolve DST edge cases
    # deterministically if a deadline happens to land in a clock transition.
    return naive.tz_localize(NY, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


@dataclass(frozen=True)
class CausalHMMConfig:
    n_components: int = 3
    covariance_type: str = "full"
    random_state: int = 42
    n_iter: int = 200
    tol: float = 0.01
    min_covar: float = 1e-3
    init_params: str = "stmc"
    params: str = "stmc"
    min_train_valid: int = 500
    sklearn_version: str = sklearn_version
    hmmlearn_version: str = ""
    numpy_version: str = np.__version__
    scipy_version: str = scipy.__version__

    def normalized(self) -> "CausalHMMConfig":
        from hmmlearn import __version__ as hmmlearn_version

        return CausalHMMConfig(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            init_params=self.init_params,
            params=self.params,
            min_train_valid=self.min_train_valid,
            sklearn_version=self.sklearn_version,
            hmmlearn_version=self.hmmlearn_version or hmmlearn_version,
            numpy_version=self.numpy_version,
            scipy_version=self.scipy_version,
        )


class CausalGaussianHMM:
    """Three-state production HMM with one consistent population scaler."""

    def __init__(self, config: CausalHMMConfig | None = None) -> None:
        self.config = (config or CausalHMMConfig()).normalized()
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            min_covar=self.config.min_covar,
            random_state=self.config.random_state,
            init_params=self.config.init_params,
            params=self.config.params,
        )
        self.fitted = False
        self.artifact_hash: str | None = None
        self.valid_feature_seconds = 0.0
        self.scaler_fit_seconds = 0.0
        self.scaler_transform_seconds = 0.0
        self.hmm_fit_seconds = 0.0

    @staticmethod
    def valid_features(rows: pd.DataFrame) -> pd.DataFrame:
        missing = set(FEATURE_SCHEMA) - set(rows.columns)
        if missing:
            raise KeyError(f"Missing causal HMM features: {sorted(missing)}")
        values = rows.loc[:, FEATURE_SCHEMA]
        matrix = values.to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(matrix).all(axis=1)
        if valid.all():
            return values
        return values.loc[valid]

    def fit(self, rows: pd.DataFrame) -> "CausalGaussianHMM":
        started = time.perf_counter()
        valid = self.valid_features(rows)
        self.valid_feature_seconds = time.perf_counter() - started
        return self._fit_valid_matrix(valid.to_numpy(dtype=np.float64, copy=False))

    def fit_matrix(
        self, values: np.ndarray, *, already_validated: bool = False
    ) -> "CausalGaussianHMM":
        """Fit directly from a compact numeric matrix without a DataFrame copy."""
        started = time.perf_counter()
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_SCHEMA):
            raise ValueError(
                f"Expected an (n, {len(FEATURE_SCHEMA)}) causal feature matrix."
            )
        if not already_validated:
            valid = np.isfinite(matrix).all(axis=1)
            if not valid.all():
                matrix = matrix[valid]
        self.valid_feature_seconds = time.perf_counter() - started
        return self._fit_valid_matrix(matrix)

    def _fit_valid_matrix(self, matrix: np.ndarray) -> "CausalGaussianHMM":
        if len(matrix) < self.config.min_train_valid:
            raise ValueError(
                f"Need {self.config.min_train_valid} valid training rows; got {len(matrix)}."
            )
        # Keep the training/inference scaling contract while timing scaler fit
        # and transform separately. Only the transformed matrix is required by
        # hmmlearn; no pandas row objects or full-history DataFrame copy is made.
        started = time.perf_counter()
        self.scaler.fit(matrix)
        self.scaler_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        z = self.scaler.transform(matrix)
        self.scaler_transform_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.model.fit(z)
        self.hmm_fit_seconds = time.perf_counter() - started
        self.fitted = True
        self.artifact_hash = self._calculate_hash()
        return self

    def _checked_row(self, row: Mapping[str, Any] | pd.Series) -> np.ndarray | None:
        try:
            values = np.asarray([row.get(name) for name in FEATURE_SCHEMA], dtype=float)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(values).all():
            return None
        return self.scaler.transform(values.reshape(1, -1))[0]

    def filter_one(
        self,
        row: Mapping[str, Any] | pd.Series,
        previous_log_posterior: np.ndarray | None,
    ) -> tuple[int, np.ndarray, np.ndarray] | None:
        if not self.fitted:
            raise RuntimeError("Causal HMM must be fitted before inference.")
        z = self._checked_row(row)
        if z is None:
            return None
        emission = self.model._compute_log_likelihood(z.reshape(1, -1))[0]
        log_transition = np.full_like(self.model.transmat_, -np.inf, dtype=float)
        positive = self.model.transmat_ > 0
        log_transition[positive] = np.log(self.model.transmat_[positive])
        if previous_log_posterior is None:
            start = np.full_like(self.model.startprob_, -np.inf, dtype=float)
            positive_start = self.model.startprob_ > 0
            start[positive_start] = np.log(self.model.startprob_[positive_start])
            posterior_log = start + emission
        else:
            previous = np.asarray(previous_log_posterior, dtype=float)
            if previous.shape != (self.config.n_components,):
                raise ValueError("Previous log posterior has the wrong shape.")
            if np.isnan(previous).any() or np.isposinf(previous).any():
                raise ValueError("Previous log posterior is invalid.")
            posterior_log = emission + logsumexp(
                previous[:, None] + log_transition, axis=0
            )
        norm = logsumexp(posterior_log)
        if not np.isfinite(norm):
            raise FloatingPointError("HMM posterior has no finite probability mass.")
        posterior_log = posterior_log - norm
        posterior = np.exp(posterior_log)
        return int(np.argmax(posterior)), posterior, posterior_log

    def filter_sequence(
        self, rows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter a fixed model through ordered rows, resetting at sequence start."""
        valid = self.valid_features(rows)
        states: list[int] = []
        probabilities: list[np.ndarray] = []
        logs: list[np.ndarray] = []
        previous = None
        for _, row in valid.iterrows():
            result = self.filter_one(row, previous)
            assert result is not None
            state, probability, previous = result
            states.append(state)
            probabilities.append(probability)
            logs.append(previous)
        return (
            np.asarray(states, dtype=np.int8),
            np.asarray(probabilities, dtype=np.float64),
            np.asarray(logs, dtype=np.float64),
        )

    def _calculate_hash(self) -> str:
        digest = sha256()
        digest.update(json.dumps(asdict(self.config), sort_keys=True).encode())
        arrays = (
            self.scaler.mean_,
            self.scaler.scale_,
            self.model.startprob_,
            self.model.transmat_,
            self.model.means_,
            self.model.covars_,
        )
        for array in arrays:
            value = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    @property
    def scaler_hash(self) -> str:
        if not self.fitted:
            raise RuntimeError("Cannot identify an unfitted causal scaler.")
        digest = sha256()
        digest.update(json.dumps({
            "feature_schema": FEATURE_SCHEMA,
            "sklearn_version": self.config.sklearn_version,
        }, sort_keys=True).encode())
        for array in (self.scaler.mean_, self.scaler.scale_, self.scaler.var_):
            value = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def raw_profiles(self) -> list[dict[str, Any]]:
        """Return diagnostic emission profiles without changing component IDs."""
        means = self.model.means_ * self.scaler.scale_[None, :] + self.scaler.mean_[None, :]
        scale_outer = np.outer(self.scaler.scale_, self.scaler.scale_)
        covariances = self.model.covars_ * scale_outer[None, :, :]
        return [
            {
                "raw_state": state,
                "mean_original_units": means[state].tolist(),
                "covariance_original_units": covariances[state].tolist(),
                "occupancy_prior": float(self.model.startprob_[state]),
                "self_transition": float(self.model.transmat_[state, state]),
                "transition_row": self.model.transmat_[state].tolist(),
            }
            for state in range(self.config.n_components)
        ]

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot checkpoint an unfitted causal HMM.")
        return {
            "version": 2,
            "config": asdict(self.config),
            "feature_schema": list(FEATURE_SCHEMA),
            "artifact_hash": self.artifact_hash,
            "scaler_hash": self.scaler_hash,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_var": self.scaler.var_.tolist(),
            "scaler_n_samples_seen": np.asarray(self.scaler.n_samples_seen_).tolist(),
            "startprob": self.model.startprob_.tolist(),
            "transmat": self.model.transmat_.tolist(),
            "means": self.model.means_.tolist(),
            "covars": self.model.covars_.tolist(),
            "monitor_iterations": int(self.model.monitor_.iter),
            "monitor_converged": bool(self.model.monitor_.converged),
            "monitor_history": list(self.model.monitor_.history),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CausalGaussianHMM":
        if int(state.get("version", 0)) != 2:
            raise ValueError("Unsupported fitted causal HMM checkpoint version.")
        if tuple(state["feature_schema"]) != FEATURE_SCHEMA:
            raise ValueError("Causal HMM checkpoint feature schema differs.")
        from hmmlearn.base import ConvergenceMonitor

        instance = cls(CausalHMMConfig(**state["config"]))
        instance.scaler.mean_ = np.asarray(state["scaler_mean"], dtype=float)
        instance.scaler.scale_ = np.asarray(state["scaler_scale"], dtype=float)
        instance.scaler.var_ = np.asarray(state["scaler_var"], dtype=float)
        instance.scaler.n_samples_seen_ = np.asarray(state["scaler_n_samples_seen"])
        if instance.scaler.n_samples_seen_.ndim == 0:
            instance.scaler.n_samples_seen_ = int(instance.scaler.n_samples_seen_)
        instance.model.n_features = len(instance.scaler.mean_)
        instance.model.startprob_ = np.asarray(state["startprob"], dtype=float)
        instance.model.transmat_ = np.asarray(state["transmat"], dtype=float)
        instance.model.means_ = np.asarray(state["means"], dtype=float)
        instance.model.covars_ = np.asarray(state["covars"], dtype=float)
        instance.model.monitor_ = ConvergenceMonitor(
            instance.model.tol, instance.model.n_iter, instance.model.verbose
        )
        instance.model.monitor_.iter = int(state.get("monitor_iterations", 0))
        for value in state.get("monitor_history", []):
            instance.model.monitor_.history.append(float(value))
        instance.fitted = True
        instance.artifact_hash = instance._calculate_hash()
        if instance.artifact_hash != state["artifact_hash"]:
            raise ValueError("Causal HMM checkpoint artifact hash is invalid.")
        if instance.scaler_hash != state["scaler_hash"]:
            raise ValueError("Causal scaler checkpoint identity is invalid.")
        return instance


class ScheduledCausalHMMStream:
    """One expanding-MR or rolling-S2R HMM stream with causal refits."""

    CHECKPOINT_VERSION = 3

    def __init__(
        self,
        *,
        schedule: str,
        refit_months: int,
        rolling_years: int | None = None,
        eligible_rth_only: bool = False,
        config: CausalHMMConfig | None = None,
    ) -> None:
        if schedule not in {"expanding", "rolling"}:
            raise ValueError("schedule must be 'expanding' or 'rolling'.")
        if refit_months <= 0 or (schedule == "rolling" and (rolling_years or 0) <= 0):
            raise ValueError("Invalid causal HMM schedule.")
        self.schedule = schedule
        self.refit_months = int(refit_months)
        self.rolling_years = rolling_years
        self.eligible_rth_only = eligible_rth_only
        self.config = (config or CausalHMMConfig()).normalized()
        history_columns = (
            tuple(dict.fromkeys((*FEATURE_SCHEMA, *S2R_FEATURES)))
            if self.eligible_rth_only
            else FEATURE_SCHEMA
        )
        self._history = _CompactObservationHistory(history_columns)
        self.anchor_timestamp: pd.Timestamp | None = None
        self.initial_fit_timestamp: pd.Timestamp | None = None
        self.last_timestamp: pd.Timestamp | None = None
        self.model: CausalGaussianHMM | None = None
        self.log_posterior: np.ndarray | None = None
        self.fit_start: pd.Timestamp | None = None
        self.fit_end_exclusive: pd.Timestamp | None = None
        self.fit_last_training_timestamp: pd.Timestamp | None = None
        self.fit_training_valid_rows = 0
        self.fit_timestamp: pd.Timestamp | None = None
        self.next_refit_timestamp: pd.Timestamp | None = None
        self.fit_count = 0
        self.training_data_hash: str | None = None
        self.s2_fitted_model: Any | None = None
        self.refit_events: list[dict[str, Any]] = []
        self.update_calls = 0
        self.invalid_feature_observations = 0
        self.forward_filter_updates = 0
        self.forward_filter_seconds = 0.0
        self.history_prune_count = 0
        self.timestamp_rejections = 0
        self.model_diagnostics: list[dict[str, Any]] = []
        self._active_model_diagnostic: dict[str, Any] | None = None
        self._last_emitted_state: int | None = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Compatibility view; production storage remains typed and compact."""
        return self._history.records()

    @history.setter
    def history(self, records: Iterable[Mapping[str, Any]]) -> None:
        self._history.replace_records(records)

    @property
    def model_hash(self) -> str | None:
        if self.model is None:
            return None
        manifest = {
            "parameters": self.model.artifact_hash,
            "schedule": self.schedule,
            "refit_months": self.refit_months,
            "rolling_years": self.rolling_years,
            "fit_start": self.fit_start.isoformat() if self.fit_start is not None else None,
            "fit_end_exclusive": self.fit_end_exclusive.isoformat() if self.fit_end_exclusive is not None else None,
            "fit_timestamp": self.fit_timestamp.isoformat() if self.fit_timestamp is not None else None,
            "next_refit_timestamp": self.next_refit_timestamp.isoformat() if self.next_refit_timestamp is not None else None,
            "s2_fitted_model": self._s2_model_state(),
        }
        if self.training_data_hash is not None:
            manifest["training_data_hash"] = self.training_data_hash
        return sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    def update(
        self,
        timestamp: Any,
        row: Mapping[str, Any],
        *,
        eligible: bool = True,
    ) -> tuple[int | None, np.ndarray | None]:
        current_time = _utc(timestamp)
        self.update_calls += 1
        if self.last_timestamp is not None and current_time <= self.last_timestamp:
            self.timestamp_rejections += 1
            raise ValueError("Causal HMM timestamps must increase strictly.")
        self.last_timestamp = current_time
        if self.anchor_timestamp is None and eligible:
            self.anchor_timestamp = current_time
            if self.schedule == "rolling" and self.rolling_years is not None:
                self.initial_fit_timestamp = add_calendar_months(
                    current_time.tz_convert(NY), self.rolling_years * 12
                )
        record = {name: row.get(name) for name in FEATURE_SCHEMA}
        if self.eligible_rth_only:
            record.update({name: row.get(name) for name in S2R_FEATURES})
        record["timestamp"] = current_time
        record["eligible"] = bool(eligible)
        if not eligible:
            return None, None

        # This is a single-observation hot path. Building a one-row DataFrame
        # and invoking pandas' column/index machinery for every market bar is
        # mathematically redundant; use the same numeric validity rule as
        # ``valid_features`` without allocating pandas objects.
        try:
            feature_values = np.asarray(
                [record.get(name) for name in FEATURE_SCHEMA], dtype=np.float64
            )
        except (TypeError, ValueError):
            feature_values = np.full(len(FEATURE_SCHEMA), np.nan)
        if not np.isfinite(feature_values).all():
            if eligible:
                self.invalid_feature_observations += 1
            return None, None
        self._append_history(current_time, record)

        due = self.next_refit_timestamp is not None and current_time >= self.next_refit_timestamp
        initial_due = self.model is None and self._initial_fit_due(current_time)
        refit_event: dict[str, Any] | None = None
        training_frame_seconds = 0.0
        training_validation_seconds = 0.0
        s2_fit_seconds = 0.0
        if due or initial_due:
            history_start, history_stop = self._training_bounds(current_time)
            fit_started = time.perf_counter()
            # MR fitting needs only the compact six-column numeric view. S2R
            # also needs its dependent quantities, so materialize the rolling
            # feature frame only for that stream.
            training = (
                self._training_frame(current_time)
                if self.eligible_rth_only
                else None
            )
            training_frame_seconds = time.perf_counter() - fit_started
            training_data_hash = self._training_fingerprint(
                history_start, history_stop
            )
            training_matrix = self._training_matrix(history_start, history_stop)
            fit_started = time.perf_counter()
            valid_train_mask = np.isfinite(training_matrix).all(axis=1)
            training_validation_seconds = time.perf_counter() - fit_started
            training_valid_rows = int(valid_train_mask.sum())
            if training_valid_rows >= self.config.min_train_valid:
                previous_hash = self.model_hash
                previous_profiles = self.model.raw_profiles() if self.model is not None else None
                previous_scaler_hash = self.model.scaler_hash if self.model is not None else None
                previous_raw_state = self._last_emitted_state
                fit_matrix = (
                    training_matrix
                    if valid_train_mask.all()
                    else training_matrix[valid_train_mask]
                )
                fresh = CausalGaussianHMM(self.config).fit_matrix(
                    fit_matrix, already_validated=True
                )
                self.model = fresh
                self.training_data_hash = training_data_hash
                self.log_posterior = None
                self.fit_timestamp = current_time
                self.fit_end_exclusive = current_time
                valid_training_timestamps = self._training_timestamps(
                    history_start, history_stop
                )[valid_train_mask]
                self.fit_last_training_timestamp = pd.Timestamp(
                    valid_training_timestamps[-1], tz="UTC"
                ) if len(valid_training_timestamps) else None
                self.fit_training_valid_rows = training_valid_rows
                self.fit_start = (
                    pd.Timestamp(
                        self._training_timestamps(history_start, history_stop)[0],
                        tz="UTC",
                    )
                    if history_stop > history_start
                    else None
                )
                self.next_refit_timestamp = add_calendar_months(
                    current_time, self.refit_months
                )
                self.fit_count += 1
                fit_started = time.perf_counter()
                self.s2_fitted_model = (
                    self._fit_s2_quantities(training, fresh)
                    if training is not None
                    else None
                )
                s2_fit_seconds = time.perf_counter() - fit_started
                self._active_model_diagnostic = {
                    "model_version": self.fit_count,
                    "model_hash": self.model_hash,
                    "fit_timestamp": current_time.isoformat(),
                    "state_occupancy": [0, 0, 0],
                    "transition_counts": [[0, 0, 0] for _ in range(3)],
                }
                self.model_diagnostics.append(self._active_model_diagnostic)
                self._last_emitted_state = None
                refit_event = {
                    "schedule": self.schedule,
                    "model_version": self.fit_count,
                    "fit_timestamp": current_time.isoformat(),
                    "training_observations": int(history_stop - history_start),
                    "training_start": self.fit_start.isoformat() if self.fit_start is not None else None,
                    "training_end_exclusive": current_time.isoformat(),
                    "last_training_timestamp": (
                        self.fit_last_training_timestamp.isoformat()
                        if self.fit_last_training_timestamp is not None else None
                    ),
                    "training_valid_rows": self.fit_training_valid_rows,
                    "training_data_hash": self.training_data_hash,
                    "first_live_timestamp": current_time.isoformat(),
                    "previous_model_hash": previous_hash,
                    "previous_scaler_hash": previous_scaler_hash,
                    "previous_raw_state_at_refit": previous_raw_state,
                    "previous_raw_component_profiles": previous_profiles,
                    "new_model_hash": self.model_hash,
                    "new_scaler_hash": fresh.scaler_hash,
                    "training_frame_seconds": training_frame_seconds,
                    "training_validation_seconds": training_validation_seconds,
                    "fit_input_validation_seconds": fresh.valid_feature_seconds,
                    "scaler_fit_seconds": fresh.scaler_fit_seconds,
                    "scaler_transform_seconds": fresh.scaler_transform_seconds,
                    "hmm_fit_seconds": fresh.hmm_fit_seconds,
                    "s2_dependent_fit_seconds": s2_fit_seconds,
                    "fit_duration_seconds": (
                        training_frame_seconds
                        + training_validation_seconds
                        + fresh.valid_feature_seconds
                        + fresh.scaler_fit_seconds
                        + fresh.scaler_transform_seconds
                        + fresh.hmm_fit_seconds
                        + s2_fit_seconds
                    ),
                    "raw_component_profiles": fresh.raw_profiles(),
                    "next_refit_timestamp": self.next_refit_timestamp.isoformat(),
                }

        if self.model is None:
            return None, None
        filter_started = time.perf_counter()
        filtered = self.model.filter_one(record, self.log_posterior)
        self.forward_filter_seconds += time.perf_counter() - filter_started
        if filtered is None:
            return None, None
        self.forward_filter_updates += 1
        state, posterior, self.log_posterior = filtered
        if self._active_model_diagnostic is not None:
            self._active_model_diagnostic["state_occupancy"][state] += 1
            if self._last_emitted_state is not None:
                self._active_model_diagnostic["transition_counts"][
                    self._last_emitted_state
                ][state] += 1
            self._last_emitted_state = state
        if refit_event is not None:
            refit_event["first_posterior"] = posterior.tolist()
            refit_event["first_raw_state"] = state
            self.refit_events.append(refit_event)
        # For a rolling stream the training-frame builder applies the exact
        # timestamp lower bound at every refit. Older rows can therefore be
        # discarded once after a successful refit, rather than rescanning the
        # entire history after every eligible observation (quadratic work).
        if refit_event is not None and self.schedule == "rolling" and self.rolling_years is not None:
            lower = add_calendar_months(current_time, -12 * self.rolling_years)
            self._prune_history(lower)
            self.history_prune_count += 1
        return state, posterior

    def _append_history(
        self, timestamp: pd.Timestamp, record: Mapping[str, Any]
    ) -> None:
        self._history.append(timestamp.value, record)

    def _prune_history(self, lower: pd.Timestamp) -> None:
        self._history.retain_from(lower.value)

    def _initial_fit_due(self, current_time: pd.Timestamp) -> bool:
        if self.schedule == "expanding":
            # MR uses its full available expanding history and starts once its
            # historical minimum of 500 valid rows can be fitted.
            return True
        assert self.initial_fit_timestamp is not None
        return current_time >= self.initial_fit_timestamp

    def _training_frame(self, live_start: pd.Timestamp) -> pd.DataFrame:
        start, stop = self._training_bounds(live_start)
        return self._history.frame(start, stop)

    def _training_matrix(self, start: int, stop: int) -> np.ndarray:
        return self._history.values[start:stop, :len(FEATURE_SCHEMA)]

    def _training_timestamps(self, start: int, stop: int) -> np.ndarray:
        return self._history.timestamps_ns[start:stop]

    def _training_fingerprint(self, start: int, stop: int) -> str:
        return self._history.fingerprint(start, stop)

    def _training_bounds(self, live_start: pd.Timestamp) -> tuple[int, int]:
        if not self._history.size:
            return 0, 0
        start = 0
        stop = int(np.searchsorted(
            self._history.timestamps_ns[: self._history.size],
            live_start.value,
            side="left",
        ))
        if self.schedule == "rolling":
            assert self.rolling_years is not None
            lower = add_calendar_months(live_start, -12 * self.rolling_years)
            start = int(np.searchsorted(
                self._history.timestamps_ns[: self._history.size],
                lower.value,
                side="left",
            ))
        return start, stop

    def _fit_s2_quantities(
        self, training: pd.DataFrame, model: CausalGaussianHMM
    ) -> Any | None:
        if not self.eligible_rth_only or training.empty:
            return None
        valid = CausalGaussianHMM.valid_features(training)
        if valid.empty:
            return None
        states, _, _ = model.filter_sequence(valid)
        from src.strategies.s2r.fitting import fit_s2_model

        missing = set(S2R_FEATURES) - set(training.columns)
        if missing:
            raise KeyError(f"S2R training rows missing fields: {sorted(missing)}")
        fitted_rows = training.loc[valid.index, S2R_FEATURES].reset_index(drop=True).copy()
        fitted_rows["hmm_state"] = states
        rows = fitted_rows.to_dict("records")
        return fit_s2_model(rows, target_state=2, tail_percent=17.5)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "schedule": self.schedule,
            "refit_months": self.refit_months,
            "rolling_years": self.rolling_years,
            "eligible_rth_only": self.eligible_rth_only,
            "config": asdict(self.config),
            "history_compact": self._history.payload(),
            "anchor_timestamp": self.anchor_timestamp.isoformat() if self.anchor_timestamp is not None else None,
            "initial_fit_timestamp": self.initial_fit_timestamp.isoformat() if self.initial_fit_timestamp is not None else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp is not None else None,
            "model": self.model.state_dict() if self.model is not None else None,
            "model_identity_hash": self.model_hash,
            "log_posterior": self.log_posterior.tolist() if self.log_posterior is not None else None,
            "fit_start": self.fit_start.isoformat() if self.fit_start is not None else None,
            "fit_end_exclusive": self.fit_end_exclusive.isoformat() if self.fit_end_exclusive is not None else None,
            "fit_last_training_timestamp": self.fit_last_training_timestamp.isoformat() if self.fit_last_training_timestamp is not None else None,
            "fit_training_valid_rows": self.fit_training_valid_rows,
            "fit_timestamp": self.fit_timestamp.isoformat() if self.fit_timestamp is not None else None,
            "next_refit_timestamp": self.next_refit_timestamp.isoformat() if self.next_refit_timestamp is not None else None,
            "fit_count": self.fit_count,
            "training_data_hash": self.training_data_hash,
            "s2_fitted_model": self._s2_model_state(),
            "refit_events": _json_value(self.refit_events),
            "update_calls": self.update_calls,
            "invalid_feature_observations": self.invalid_feature_observations,
            "forward_filter_updates": self.forward_filter_updates,
            "forward_filter_seconds": self.forward_filter_seconds,
            "history_prune_count": self.history_prune_count,
            "timestamp_rejections": self.timestamp_rejections,
            "model_diagnostics": _json_value(self.model_diagnostics),
            "last_emitted_state": self._last_emitted_state,
        }

    def _s2_model_state(self) -> dict[str, Any] | None:
        if self.s2_fitted_model is None:
            return None
        return {
            "thresholds": dict(self.s2_fitted_model.signal_model.thresholds),
            "scales": dict(self.s2_fitted_model.signal_model.scales),
            "volatility_reference": list(self.s2_fitted_model.volatility_reference),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ScheduledCausalHMMStream":
        checkpoint_version = int(state["version"])
        if checkpoint_version not in {2, cls.CHECKPOINT_VERSION}:
            raise ValueError("Unsupported causal HMM checkpoint version.")
        stream = cls(
            schedule=state["schedule"],
            refit_months=int(state["refit_months"]),
            rolling_years=state["rolling_years"],
            eligible_rth_only=bool(state["eligible_rth_only"]),
            config=CausalHMMConfig(**state["config"]),
        )
        if checkpoint_version == 2:
            stream.history = list(state["history"])
        else:
            compact = _CompactObservationHistory.from_payload(state["history_compact"])
            expected_columns = stream._history.columns
            if compact.columns != expected_columns:
                raise ValueError("Causal compact-history schema differs.")
            stream._history = compact
        for attr in (
            "anchor_timestamp", "last_timestamp", "fit_start", "fit_end_exclusive",
            "fit_last_training_timestamp", "initial_fit_timestamp",
            "fit_timestamp", "next_refit_timestamp",
        ):
            value = state.get(attr)
            setattr(stream, attr, _utc(value) if value is not None else None)
        if state.get("model") is not None:
            stream.model = CausalGaussianHMM.from_state_dict(state["model"])
        if state.get("log_posterior") is not None:
            stream.log_posterior = np.asarray(state["log_posterior"], dtype=float)
        stream.fit_count = int(state.get("fit_count", 0))
        stream.training_data_hash = state.get("training_data_hash")
        stream.fit_training_valid_rows = int(state.get("fit_training_valid_rows", 0))
        stream.refit_events = list(state.get("refit_events", []))
        for name in (
            "update_calls", "invalid_feature_observations", "forward_filter_updates",
            "history_prune_count", "timestamp_rejections",
        ):
            setattr(stream, name, int(state.get(name, 0)))
        stream.forward_filter_seconds = float(state.get("forward_filter_seconds", 0.0))
        stream.model_diagnostics = list(state.get("model_diagnostics", []))
        stream._active_model_diagnostic = (
            stream.model_diagnostics[-1] if stream.model_diagnostics else None
        )
        last_emitted = state.get("last_emitted_state")
        stream._last_emitted_state = int(last_emitted) if last_emitted is not None else None
        s2 = state.get("s2_fitted_model")
        if s2 is not None:
            from src.strategies.s2r.fitting import S2FittedModel
            from src.strategies.s2r.signal import S2SignalModel

            stream.s2_fitted_model = S2FittedModel(
                signal_model=S2SignalModel(
                    thresholds=s2["thresholds"], scales=s2["scales"]
                ),
                volatility_reference=tuple(s2["volatility_reference"]),
            )
        if stream.model_hash != state.get("model_identity_hash"):
            raise ValueError("Causal HMM fit-interval identity hash is invalid.")
        return stream


class ScheduledRawStateProvider:
    """MR expanding HMM plus independent S2R rolling HMM; IDs stay raw."""

    TEST_ONLY = False

    def __init__(self, *, config: CausalHMMConfig | None = None) -> None:
        config = (config or CausalHMMConfig()).normalized()
        self.mr = ScheduledCausalHMMStream(
            schedule="expanding", refit_months=4, config=config
        )
        self.s2r = ScheduledCausalHMMStream(
            schedule="rolling",
            refit_months=3,
            rolling_years=2,
            eligible_rth_only=True,
            config=config,
        )

    def state_for(self, timestamp: Any, row: Mapping[str, Any]) -> int | None:
        state, _ = self.mr.update(timestamp, row)
        return state

    def s2r_state_for(
        self, timestamp: Any, row: Mapping[str, Any], *, is_rth: bool
    ) -> int | None:
        state, _ = self.s2r.update(timestamp, row, eligible=is_rth)
        return state

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "provider": "ScheduledRawStateProvider",
            "feature_schema": list(FEATURE_SCHEMA),
            "mr": self.mr.state_dict(),
            "s2r": self.s2r.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ScheduledRawStateProvider":
        if int(state.get("version", 0)) != 2 or state.get("provider") != "ScheduledRawStateProvider" or tuple(state["feature_schema"]) != FEATURE_SCHEMA:
            raise ValueError("Unsupported raw-state HMM provider checkpoint.")
        instance = cls.__new__(cls)
        instance.mr = ScheduledCausalHMMStream.from_state_dict(state["mr"])
        instance.s2r = ScheduledCausalHMMStream.from_state_dict(state["s2r"])
        return instance
