from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeanReversionCandidate(Enum):
    """Frozen Mean Reversion research candidates."""

    MRS2 = "MRS2"
    MRL1 = "MRL1"
    MRL2 = "MRL2"


@dataclass(frozen=True)
class MeanReversionConfig:
    """
    Frozen configuration for one Mean Reversion candidate.

    Parameters originate from the frozen research candidates.

    TP / SL are expressed in MNQ price points.
    Horizon is expressed in bars.
    Z-score is a threshold magnitude.
    """

    candidate_id: str
    name: str
    side: str

    hmm_state: int

    volatility_low: float
    volatility_high: float

    zscore_threshold: float

    target_points: float
    stop_points: float
    horizon_bars: int

    @property
    def rr(self) -> float:
        """Return target / stop ratio."""

        if self.stop_points <= 0:
            raise ValueError("stop_points must be positive.")

        return self.target_points / self.stop_points


MRS2_CONFIG = MeanReversionConfig(
    candidate_id="C01",
    name="MRS2",
    side="SHORT",
    hmm_state=2,
    volatility_low=80.0,
    volatility_high=100.0,
    zscore_threshold=2.0,
    target_points=5.0,
    stop_points=2.0,
    horizon_bars=5,
)


MRL1_CONFIG = MeanReversionConfig(
    candidate_id="C02",
    name="MRL1",
    side="LONG",
    hmm_state=1,
    volatility_low=20.0,
    volatility_high=40.0,
    zscore_threshold=2.5,
    target_points=5.0,
    stop_points=2.0,
    horizon_bars=20,
)


MRL2_CONFIG = MeanReversionConfig(
    candidate_id="C06",
    name="MRL2",
    side="LONG",
    hmm_state=2,
    volatility_low=60.0,
    volatility_high=80.0,
    zscore_threshold=3.5,
    target_points=5.0,
    stop_points=2.0,
    horizon_bars=2,
)


FROZEN_CONFIGS: dict[MeanReversionCandidate, MeanReversionConfig] = {
    MeanReversionCandidate.MRS2: MRS2_CONFIG,
    MeanReversionCandidate.MRL1: MRL1_CONFIG,
    MeanReversionCandidate.MRL2: MRL2_CONFIG,
}
