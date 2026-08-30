from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Iterable, Mapping, Sequence

import numpy as np

from .signal import BASE_FEATURES, S2SignalModel


@dataclass(frozen=True)
class S2FittedModel:
    """
    Complete frozen S2 model fitted from TRAIN data only.

    The fitted thresholds, quality scales, and volatility reference are
    immutable after construction.
    """

    signal_model: S2SignalModel
    volatility_reference: tuple[float, ...]

    def transform_volatility(
        self,
        value: float | None,
    ) -> float | None:
        """
        Transform one realized-volatility observation using TRAIN only.

        Matches the benchmark's np.searchsorted(..., side="right")
        semantics exactly.
        """

        if value is None:
            return None

        if not isinstance(value, (int, float)):
            return None

        value = float(value)

        if isnan(value) or np.isinf(value):
            return None

        if not self.volatility_reference:
            return None

        position = np.searchsorted(
            np.asarray(self.volatility_reference),
            value,
            side="right",
        )

        return float(position / len(self.volatility_reference))


def _clean_values(values: Iterable[float]) -> list[float]:
    """Remove NaN and infinite values."""

    cleaned: list[float] = []

    for value in values:
        if not isinstance(value, (int, float)):
            continue

        value = float(value)

        if isnan(value) or np.isinf(value):
            continue

        cleaned.append(value)

    return cleaned


def _quantile(values: Sequence[float], q: float) -> float:
    """
    Calculate the quantile using NumPy's default linear interpolation.

    This matches pandas Series.quantile() for the benchmark's numeric data.
    """

    if not values:
        return float("nan")

    return float(np.quantile(np.asarray(values, dtype=float), q))


def fit_s2_signal_model(
    train_rows: Iterable[Mapping[str, float]],
    *,
    target_state: int = 2,
    tail_percent: float = 17.5,
) -> S2SignalModel:
    """
    Fit S2 feature thresholds and quality scales from TRAIN only.

    Rows outside target_state are ignored.
    """

    state_rows = [row for row in train_rows if row.get("hmm_state") == target_state]

    thresholds: dict[str, float] = {}
    scales: dict[str, float] = {}

    q = tail_percent / 100.0

    for feature in BASE_FEATURES:
        values = _clean_values(row[feature] for row in state_rows if feature in row)

        threshold = _quantile(values, q)
        extreme = _quantile(values, 0.05)

        scale = threshold - extreme

        if scale <= 0:
            scale = float("nan")

        thresholds[feature] = threshold
        scales[feature] = float(scale)

    return S2SignalModel(
        thresholds=thresholds,
        scales=scales,
    )


def fit_volatility_reference(
    train_rows: Iterable[Mapping[str, float]],
) -> tuple[float, ...]:
    """
    Build the frozen volatility reference distribution from TRAIN only.
    """

    values = _clean_values(
        row["realized_vol_30"] for row in train_rows if "realized_vol_30" in row
    )

    values.sort()

    return tuple(values)


def fit_s2_model(
    train_rows: Iterable[Mapping[str, float]],
    *,
    target_state: int = 2,
    tail_percent: float = 17.5,
) -> S2FittedModel:
    """
    Fit the complete S2 model from TRAIN data only.

    The returned model contains everything required to evaluate OOS
    observations without refitting.
    """

    rows = list(train_rows)

    signal_model = fit_s2_signal_model(
        rows,
        target_state=target_state,
        tail_percent=tail_percent,
    )

    volatility_reference = fit_volatility_reference(rows)

    return S2FittedModel(
        signal_model=signal_model,
        volatility_reference=volatility_reference,
    )
