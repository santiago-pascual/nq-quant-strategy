from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .regime import HMM_FEATURES, VolatilityRegimeModel


@dataclass(frozen=True, order=True)
class ResearchHMMWindow:
    window: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.oos_start)
        end = pd.Timestamp(self.oos_end)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Research HMM window boundaries must be timezone-aware.")
        if end < start:
            raise ValueError("Research HMM window end precedes its start.")
        if self.window <= 0:
            raise ValueError("Research HMM window number must be positive.")
        object.__setattr__(self, "oos_start", start.tz_convert("UTC"))
        object.__setattr__(self, "oos_end", end.tz_convert("UTC"))


def build_research_hmm_windows(
    rth_features: pd.DataFrame,
    *,
    n_windows: int = 22,
    timestamp_column: str = "timestamp",
    event_column: str = "zscore_30",
) -> tuple[ResearchHMMWindow, ...]:
    """Reproduce Research 07 row blocks and Research 08B event bounds."""
    schedule = build_research_hmm_schedule_metadata(
        rth_features,
        n_windows=n_windows,
        timestamp_column=timestamp_column,
        event_column=event_column,
    )
    return tuple(
        ResearchHMMWindow(
            window=int(row.window),
            oos_start=row.oos_start,
            oos_end=row.oos_end,
        )
        for row in schedule.itertuples(index=False)
    )


def build_research_hmm_schedule_metadata(
    rth_features: pd.DataFrame,
    *,
    n_windows: int = 22,
    timestamp_column: str = "timestamp",
    event_column: str = "zscore_30",
) -> pd.DataFrame:
    """Build a serializable boundary manifest using the Research 07 split."""
    if n_windows <= 0:
        raise ValueError("n_windows must be positive.")
    if timestamp_column not in rth_features or event_column not in rth_features:
        raise KeyError(f"Required columns: {timestamp_column!r}, {event_column!r}")
    ordered = rth_features.sort_values(timestamp_column, kind="mergesort").reset_index(
        drop=True
    )
    timestamps = pd.to_datetime(ordered[timestamp_column], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Research window input contains duplicate timestamps.")
    if timestamps.empty:
        raise ValueError("Research window input is empty.")
    blocks = np.array_split(np.arange(len(ordered), dtype=np.int64), n_windows)
    metadata: list[dict[str, object]] = []
    for number, indices in enumerate(blocks, start=1):
        if not len(indices):
            raise ValueError(
                f"Research 07 window {number} has no rows in the supplied universe."
            )
        events = indices[pd.to_numeric(
            ordered.iloc[indices][event_column], errors="coerce"
        ).notna().to_numpy()]
        if not len(events):
            raise ValueError(
                f"Research 07 window {number} has no valid event boundaries."
            )
        metadata.append(
            {
                "window": number,
                "rth_universe_start": timestamps.iloc[0],
                "rth_universe_end": timestamps.iloc[-1],
                "rth_block_start": timestamps.iloc[int(indices[0])],
                "rth_block_end": timestamps.iloc[int(indices[-1])],
                "training_start": timestamps.iloc[0],
                "training_end_exclusive": timestamps.iloc[int(events[0])],
                "oos_start": timestamps.iloc[int(events[0])],
                "oos_end": timestamps.iloc[int(events[-1])],
            }
        )
    return pd.DataFrame(metadata)


def load_frozen_research_hmm_windows(
    schedule_path: str,
    *,
    expected_windows: int = 22,
) -> tuple[ResearchHMMWindow, ...]:
    """Load and validate the persisted Research 07/08B window boundaries."""
    required = {
        "window",
        "rth_universe_start",
        "rth_universe_end",
        "rth_block_start",
        "rth_block_end",
        "training_start",
        "training_end_exclusive",
        "oos_start",
        "oos_end",
    }
    schedule = pd.read_csv(schedule_path)
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(
            f"Frozen HMM schedule is missing columns: {sorted(missing)}"
        )
    if len(schedule) != expected_windows:
        raise ValueError(
            f"Frozen HMM schedule must contain exactly {expected_windows} windows; "
            f"found {len(schedule)}."
        )
    schedule["window"] = pd.to_numeric(schedule["window"], errors="raise").astype(int)
    if schedule["window"].tolist() != list(range(1, expected_windows + 1)):
        raise ValueError("Frozen HMM schedule window IDs must be 1..N in order.")
    timestamp_columns = required - {"window"}
    for column in timestamp_columns:
        schedule[column] = pd.to_datetime(schedule[column], utc=True, errors="raise")
    if (
        schedule["training_start"].nunique() != 1
        or schedule["rth_universe_start"].nunique() != 1
        or schedule["rth_universe_end"].nunique() != 1
    ):
        raise ValueError("Frozen HMM schedule has inconsistent universe boundaries.")
    if not (schedule["training_start"] == schedule["rth_universe_start"]).all():
        raise ValueError("Training start must match the frozen RTH universe start.")
    if (
        schedule["rth_block_start"].iloc[0] != schedule["rth_universe_start"].iloc[0]
        or schedule["rth_block_end"].iloc[-1] != schedule["rth_universe_end"].iloc[-1]
    ):
        raise ValueError("Frozen RTH blocks do not span the declared universe.")
    if not (
        schedule["training_end_exclusive"] == schedule["oos_start"]
    ).all():
        raise ValueError("Training end must be the exclusive OOS start.")
    if (schedule["rth_block_start"] > schedule["rth_block_end"]).any():
        raise ValueError("Frozen HMM schedule contains an inverted RTH block.")
    if (schedule["oos_start"] > schedule["oos_end"]).any():
        raise ValueError("Frozen HMM schedule contains an inverted OOS window.")
    if (
        (schedule["oos_start"] < schedule["rth_block_start"])
        | (schedule["oos_end"] > schedule["rth_block_end"])
    ).any():
        raise ValueError("Frozen OOS bounds fall outside their Research 07 block.")
    if (
        schedule["oos_start"].iloc[1:].reset_index(drop=True)
        <= schedule["oos_end"].iloc[:-1].reset_index(drop=True)
    ).any():
        raise ValueError("Frozen HMM OOS windows overlap or are out of order.")
    if (
        schedule["rth_block_start"].iloc[1:].reset_index(drop=True)
        <= schedule["rth_block_end"].iloc[:-1].reset_index(drop=True)
    ).any():
        raise ValueError("Frozen Research 07 RTH blocks overlap or are out of order.")
    return tuple(
        ResearchHMMWindow(
            window=int(row.window),
            oos_start=row.oos_start,
            oos_end=row.oos_end,
        )
        for row in schedule.itertuples(index=False)
    )


def research_hmm_window_for_timestamp(
    windows: Iterable[ResearchHMMWindow],
    timestamp: object,
) -> ResearchHMMWindow | None:
    current = pd.Timestamp(timestamp)
    if current.tzinfo is None:
        raise ValueError("Research HMM timestamp must be timezone-aware.")
    current = current.tz_convert("UTC")
    return next(
        (
            window
            for window in windows
            if window.oos_start <= current <= window.oos_end
        ),
        None,
    )


ModelFactory = Callable[[], VolatilityRegimeModel]


class WindowedLiveHMM:
    """Fit and freeze one HMM per fixed research OOS window.

    Each window's fit uses only observations strictly before its OOS start.
    In-window states are forward-filtered one completed row at a time; raw
    GaussianHMM component IDs are returned without remapping.
    """

    def __init__(
        self,
        windows: Iterable[ResearchHMMWindow],
        *,
        min_train_valid: int = 500,
        model_factory: ModelFactory | None = None,
    ) -> None:
        ordered = tuple(sorted(windows, key=lambda item: item.window))
        if not ordered:
            raise ValueError("At least one research HMM window is required.")
        if len({item.window for item in ordered}) != len(ordered):
            raise ValueError("Research HMM window numbers must be unique.")
        if any(
            current.oos_start <= previous.oos_end
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError("Research HMM windows must be chronological and disjoint.")
        if min_train_valid <= 0:
            raise ValueError("min_train_valid must be positive.")
        self.windows = ordered
        self.min_train_valid = min_train_valid
        self._model_factory = model_factory or (
            lambda: VolatilityRegimeModel(n_states=3, random_state=42, n_iter=200)
        )
        self._history: list[dict[str, object]] = []
        self._models: dict[int, VolatilityRegimeModel] = {}
        self._failed_windows: set[int] = set()
        self._posteriors: dict[int, np.ndarray] = {}
        self._last_timestamp: pd.Timestamp | None = None

    @property
    def models(self) -> dict[int, VolatilityRegimeModel]:
        return dict(self._models)

    @property
    def fitted_windows(self) -> tuple[int, ...]:
        return tuple(sorted(self._models))

    def prime_history(self, features: pd.DataFrame) -> None:
        """Load pre-replay training history without decoding prior OOS rows."""
        if self._last_timestamp is not None or self._history or self._models:
            raise RuntimeError("Windowed HMM history has already been initialized.")
        required = {"timestamp", *HMM_FEATURES}
        missing = required - set(features.columns)
        if missing:
            raise KeyError(f"HMM history missing columns: {sorted(missing)}")
        history = features.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
        if not history["timestamp"].is_monotonic_increasing:
            raise ValueError("HMM history timestamps must be chronological.")
        self._history = history.to_dict(orient="records")
        if self._history:
            self._last_timestamp = pd.Timestamp(self._history[-1]["timestamp"])

    def update(self, feature_row: pd.Series | dict[str, object]) -> int | None:
        row = dict(feature_row)
        if "timestamp" not in row:
            raise KeyError("Live HMM feature row must contain timestamp.")
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError("Live HMM timestamp must be timezone-aware.")
        timestamp = timestamp.tz_convert("UTC")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("Live HMM timestamps must increase strictly.")
        self._last_timestamp = timestamp
        row["timestamp"] = timestamp

        window = research_hmm_window_for_timestamp(self.windows, timestamp)
        state: int | None = None
        if window is not None and window.window not in self._failed_windows:
            model = self._models.get(window.window)
            if model is None:
                training = pd.DataFrame(self._history)
                training = training.loc[
                    pd.to_datetime(training["timestamp"], utc=True)
                    < window.oos_start
                ]
                valid = training[HMM_FEATURES].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if len(valid) >= self.min_train_valid:
                    model = self._model_factory()
                    model.fit(training)
                    self._models[window.window] = model
                else:
                    self._failed_windows.add(window.window)
            if model is not None:
                current = pd.DataFrame([{name: row.get(name) for name in HMM_FEATURES}])
                valid_current = (
                    current.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
                )
                if bool(valid_current.iloc[0]):
                    previous = self._posteriors.get(window.window)
                    state, posterior = model.predict_causal_state(
                        current,
                        previous_probabilities=previous,
                    )
                    self._posteriors[window.window] = posterior

        self._history.append(
            {
                "timestamp": timestamp,
                **{name: value for name, value in row.items() if name != "timestamp"},
            }
        )
        return state

    def training_frame(self, window: int) -> pd.DataFrame:
        """Return only recorded observations strictly before a window start."""
        definition = next(
            (item for item in self.windows if item.window == window),
            None,
        )
        if definition is None:
            raise KeyError(f"Unknown research HMM window: {window}")
        history = pd.DataFrame(self._history)
        if history.empty:
            return history
        timestamps = pd.to_datetime(history["timestamp"], utc=True)
        return history.loc[timestamps < definition.oos_start].reset_index(drop=True)
