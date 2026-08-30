from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Mapping


BASE_FEATURES = (
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
)


@dataclass(frozen=True)
class S2SignalModel:
    """
    Frozen S2 signal model.

    Thresholds and scales must be fitted exclusively from TRAIN data.
    Once fitted, the model is deterministic and can be applied to OOS/live
    observations without refitting.
    """

    thresholds: Mapping[str, float]
    scales: Mapping[str, float]

    def calculate_quality(
        self,
        features: Mapping[str, float],
    ) -> float | None:
        """
        Calculate the S2 quality score for one observation.

        Returns None when any required feature or scale is invalid.
        """

        scores: list[float] = []

        for feature in BASE_FEATURES:
            value = features.get(feature)
            threshold = self.thresholds.get(feature)
            scale = self.scales.get(feature)

            if value is None or threshold is None or scale is None:
                return None

            if not all(
                isinstance(x, (int, float)) and not isnan(float(x))
                for x in (value, threshold, scale)
            ):
                return None

            if scale <= 0:
                return None

            score = (threshold - value) / scale
            score = max(0.0, min(1.0, score))
            scores.append(score)

        return sum(scores) / len(scores)

    def base_signal(
        self,
        features: Mapping[str, float],
    ) -> bool:
        """Evaluate the frozen S2 feature-threshold signal."""

        for feature in BASE_FEATURES:
            value = features.get(feature)
            threshold = self.thresholds.get(feature)

            if value is None or threshold is None:
                return False

            if not isinstance(value, (int, float)):
                return False

            if isnan(float(value)) or isnan(float(threshold)):
                return False

            if value > threshold:
                return False

        return True


@dataclass(frozen=True)
class S2SignalRule:
    """Complete frozen S2 entry rule."""

    target_state: int = 2
    quality_threshold: float = 0.75
    volatility_low: float = 0.40
    volatility_high: float = 0.60

    def qualifies(
        self,
        *,
        hmm_state: int,
        quality: float | None,
        volatility_percentile: float | None,
    ) -> bool:
        """Return whether the observation qualifies for an S2 entry."""

        if hmm_state != self.target_state:
            return False

        if quality is None:
            return False

        if volatility_percentile is None:
            return False

        if not isinstance(quality, (int, float)):
            return False

        if not isinstance(volatility_percentile, (int, float)):
            return False

        if isnan(float(quality)) or isnan(float(volatility_percentile)):
            return False

        if quality < self.quality_threshold:
            return False

        return self.volatility_low <= volatility_percentile < self.volatility_high
