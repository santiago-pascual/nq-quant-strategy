from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.models.regime import HMM_FEATURES
from src.models.windowed_regime import ResearchHMMWindow, WindowedLiveHMM
from src.paper.market_context import CausalMarketContext, MarketContextConfig


class FixedStateModel:
    def fit(self, frame: pd.DataFrame):
        self.training_rows = len(frame)
        return self

    def predict_causal_state(self, frame, *, previous_probabilities=None):
        return 2, np.asarray([0.0, 0.0, 1.0])

    def predict_states(self, frame: pd.DataFrame) -> pd.Series:
        valid = frame[HMM_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
        return pd.Series(2, index=valid.index, name="hmm_state")


def test_s2r_parameters_are_fitted_from_the_windows_frozen_hmm_labels():
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    history = pd.DataFrame(
        {
            "timestamp": [start - timedelta(minutes=50 - i) for i in range(50)],
            **{feature: np.linspace(-2.0, 2.0, 50) for feature in HMM_FEATURES},
            "past_return_30": np.linspace(-5.0, 5.0, 50),
            "directional_pressure_30": np.linspace(-1.0, 1.0, 50),
            "close_location_30": np.linspace(0.0, 1.0, 50),
            "normalized_momentum_30": np.linspace(-3.0, 3.0, 50),
            "realized_vol_30": np.linspace(0.1, 2.0, 50),
        }
    )
    window = ResearchHMMWindow(
        1,
        pd.Timestamp(start),
        pd.Timestamp(start + timedelta(days=1)),
    )
    manager = WindowedLiveHMM(
        [window],
        min_train_valid=2,
        model_factory=FixedStateModel,
    )
    manager.prime_history(history)
    current = history.iloc[-1].to_dict()
    current["timestamp"] = pd.Timestamp(start)
    assert manager.update(current) == 2

    context = CausalMarketContext(
        MarketContextConfig(hmm_min_train_valid=2)
    )
    context._windowed_hmm = manager
    fitted = context.s2_model_for_window(1)

    assert manager.fitted_windows == (1,)
    assert fitted.volatility_reference == tuple(
        sorted(history["realized_vol_30"].tolist())
    )
    assert all(np.isfinite(value) for value in fitted.signal_model.thresholds.values())
