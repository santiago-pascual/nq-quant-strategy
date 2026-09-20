from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeanReversionCandidate(Enum):
    """Frozen Mean Reversion production candidates."""

    MRS2 = "MRS2"
    MRL1 = "MRL1"


@dataclass(frozen=True)
class MeanReversionConfig:
    """
    Frozen configuration for one Mean Reversion candidate.

    Parameters originate from the final validated research branch.

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


# =============================================================================
# FROZEN MRS2
# =============================================================================
#
# SHORT
# HMM State 2
# VOL 80-100
# Z >= +2.0
# TP 27.5
# SL 25.0
# RR 1.10
# Horizon 30 bars
#
# Validated through:
#   08AA — full robustness suite
#   08AB — temporal robustness
#   08AC — MAE/MFE failure analysis
#   08AD — Monte Carlo/bootstrap
#   08AE — parameter perturbation
#   08AF — cost/slippage
# =============================================================================

MRS2_CONFIG = MeanReversionConfig(
    candidate_id="MRS2_NEW",
    name="MRS2",
    side="SHORT",
    hmm_state=2,
    volatility_low=80.0,
    volatility_high=100.0,
    zscore_threshold=2.0,
    target_points=27.5,
    stop_points=25.0,
    horizon_bars=30,
)


# =============================================================================
# FROZEN MRL1
# =============================================================================
#
# LONG
# HMM State 1
# VOL 20-40
# Z <= -2.5
# TP 25.0
# SL 37.5
# RR 0.6667
# Horizon 8 bars
#
# Validated through:
#   08AA — full robustness suite
#   08AB — temporal robustness
#   08AC — MAE/MFE failure analysis
#   08AD — Monte Carlo/bootstrap
#   08AE — parameter perturbation
#   08AF — cost/slippage
# =============================================================================

MRL1_CONFIG = MeanReversionConfig(
    candidate_id="MRL1_NEW",
    name="MRL1",
    side="LONG",
    hmm_state=1,
    volatility_low=20.0,
    volatility_high=40.0,
    zscore_threshold=2.5,
    target_points=25.0,
    stop_points=37.5,
    horizon_bars=8,
)


# =============================================================================
# FROZEN CONFIGURATIONS
# =============================================================================

FROZEN_CONFIGS: dict[MeanReversionCandidate, MeanReversionConfig] = {
    MeanReversionCandidate.MRS2: MRS2_CONFIG,
    MeanReversionCandidate.MRL1: MRL1_CONFIG,
}
