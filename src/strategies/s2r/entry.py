from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class S2EntryContext:
    """
    Precomputed information required by the frozen S2 entry rule.

    Feature engineering and HMM inference happen outside this component.
    """

    hmm_state: int
    quality: float
    volatility_percentile: float


@dataclass(frozen=True)
class S2EntryRule:
    """
    Frozen S2 short-entry rule.

    The rule is deliberately pure: it receives already-computed model
    outputs and determines whether the S2 entry conditions are satisfied.
    """

    target_state: int = 2
    quality_threshold: float = 0.75
    volatility_low: float = 0.40
    volatility_high: float = 0.60

    def qualifies(self, context: S2EntryContext) -> bool:
        """Return True when all frozen S2 entry conditions are satisfied."""

        if context.hmm_state != self.target_state:
            return False

        if context.quality < self.quality_threshold:
            return False

        if not (
            self.volatility_low <= context.volatility_percentile < self.volatility_high
        ):
            return False

        return True
