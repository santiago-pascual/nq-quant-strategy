from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.models.regime import HMM_FEATURES
from src.models.windowed_regime import (
    ResearchHMMWindow,
    WindowedLiveHMM,
    build_research_hmm_windows,
)
from src.strategies.base import StrategySignal
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy


class FixedRawLabelModel:
    fit_calls = 0

    def __init__(self, label: int) -> None:
        self.label = label
        self.training_timestamps: list[pd.Timestamp] = []

    def fit(self, frame: pd.DataFrame) -> FixedRawLabelModel:
        type(self).fit_calls += 1
        self.training_timestamps = list(pd.to_datetime(frame["timestamp"], utc=True))
        return self

    def predict_causal_state(
        self,
        frame: pd.DataFrame,
        *,
        previous_probabilities: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        return self.label, np.eye(3, dtype=float)[self.label]


def _row(timestamp: datetime, value: float = 1.0) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        **{feature: value for feature in HMM_FEATURES},
    }


def _window(
    number: int,
    start: datetime,
    end: datetime,
) -> ResearchHMMWindow:
    return ResearchHMMWindow(number, pd.Timestamp(start), pd.Timestamp(end))


def test_window_model_fits_once_and_training_stops_before_oos():
    FixedRawLabelModel.fit_calls = 0
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=2)
    model = FixedRawLabelModel(2)
    manager = WindowedLiveHMM(
        [_window(1, start, end)],
        min_train_valid=2,
        model_factory=lambda: model,
    )

    manager.update(_row(start - timedelta(minutes=2)))
    manager.update(_row(start - timedelta(minutes=1)))
    assert manager.update(_row(start)) == 2
    assert manager.update(_row(start + timedelta(minutes=1))) == 2
    assert FixedRawLabelModel.fit_calls == 1
    assert max(model.training_timestamps) < pd.Timestamp(start)


def test_future_rows_do_not_change_current_causal_state():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=3)
    prefix = [_row(start - timedelta(minutes=2)), _row(start - timedelta(minutes=1))]
    current = _row(start)

    def evaluate(future_rows: list[dict[str, object]]) -> int | None:
        manager = WindowedLiveHMM(
            [_window(1, start, end)],
            min_train_valid=2,
            model_factory=lambda: FixedRawLabelModel(1),
        )
        for item in prefix:
            manager.update(item)
        state = manager.update(current)
        for item in future_rows:
            manager.update(item)
        return state

    future = [_row(start + timedelta(minutes=i), value=float(i)) for i in range(1, 5)]
    assert evaluate([]) == evaluate(future) == 1


def test_window_boundary_selects_new_model_without_remapping_labels():
    first = datetime(2024, 1, 1, tzinfo=timezone.utc)
    second = first + timedelta(days=2)
    windows = [
        _window(1, first, first + timedelta(days=1)),
        _window(2, second, second + timedelta(days=1)),
    ]
    labels = iter((1, 2))
    models: list[FixedRawLabelModel] = []

    def factory() -> FixedRawLabelModel:
        model = FixedRawLabelModel(next(labels))
        models.append(model)
        return model

    manager = WindowedLiveHMM(windows, min_train_valid=2, model_factory=factory)
    manager.update(_row(first - timedelta(minutes=2)))
    manager.update(_row(first - timedelta(minutes=1)))
    assert manager.update(_row(first)) == 1
    manager.update(_row(first + timedelta(days=1)))
    manager.update(_row(second - timedelta(minutes=1)))
    assert manager.update(_row(second)) == 2
    assert manager.fitted_windows == (1, 2)
    assert len(models) == 2


def test_research_window_builder_uses_equal_row_blocks_and_valid_event_bounds():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC")
    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "zscore_30": [np.nan, 0.0, 1.0, 2.0, 3.0, np.nan, 1.0, 2.0, 3.0, 4.0],
        }
    )
    windows = build_research_hmm_windows(
        features,
        n_windows=2,
        event_column="zscore_30",
    )
    assert [(item.window, item.oos_start, item.oos_end) for item in windows] == [
        (1, timestamps[1], timestamps[4]),
        (2, timestamps[6], timestamps[9]),
    ]


def test_mrl1_and_mrs2_gates_use_their_unmodified_raw_state_ids():
    mrl1 = MeanReversionStrategy(MRL1_CONFIG)
    mrs2 = MeanReversionStrategy(MRS2_CONFIG)
    mrl1_context = {"hmm_state": 1, "vol_percentile": 30.0, "zscore": -2.5}
    mrs2_context = {"hmm_state": 2, "vol_percentile": 90.0, "zscore": 2.0}

    assert mrl1.generate_signal(mrl1_context) is StrategySignal.LONG
    assert mrl1.generate_signal({**mrl1_context, "hmm_state": 2}) is StrategySignal.FLAT
    assert mrs2.generate_signal(mrs2_context) is StrategySignal.SHORT
    assert mrs2.generate_signal({**mrs2_context, "hmm_state": 1}) is StrategySignal.FLAT
