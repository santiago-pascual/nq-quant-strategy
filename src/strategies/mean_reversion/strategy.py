from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import (
    BaseStrategy,
    StrategySignal,
)
from src.strategies.mean_reversion.config import (
    FROZEN_CONFIGS,
    MeanReversionCandidate,
    MeanReversionConfig,
)


class MeanReversionStrategy(BaseStrategy):
    """
    Production Mean Reversion strategy.

    The strategy is parameterized by one frozen candidate configuration.

    Signal generation is intentionally limited to the strategy's entry
    conditions. Risk geometry such as stop distance is exposed separately
    through get_risk_stop_price().
    """

    def __init__(
        self,
        config: MeanReversionConfig,
    ) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def candidate_id(self) -> str:
        return self.config.candidate_id

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        """
        Generate the Mean Reversion directional signal.

        MRS2:
            SHORT
            HMM state 2
            volatility percentile [80, 100)
            z-score >= +2.0

        MRL1:
            LONG
            HMM state 1
            volatility percentile [20, 40)
            z-score <= -2.5
        """

        hmm_state = market_data.get("hmm_state")
        vol_percentile = market_data.get("vol_percentile")
        zscore = market_data.get("zscore")

        if hmm_state is None:
            return StrategySignal.FLAT

        if vol_percentile is None:
            return StrategySignal.FLAT

        if zscore is None:
            return StrategySignal.FLAT

        try:
            hmm_state_int = int(hmm_state)
            volatility = float(vol_percentile)
            zscore_value = float(zscore)
        except (TypeError, ValueError):
            return StrategySignal.FLAT

        if (
            self.config.candidate_id
            == FROZEN_CONFIGS[MeanReversionCandidate.MRS2].candidate_id
        ):
            if (
                hmm_state_int == self.config.hmm_state
                and self.config.volatility_low
                <= volatility
                < self.config.volatility_high
                and zscore_value >= self.config.zscore_threshold
            ):
                return StrategySignal.SHORT

            return StrategySignal.FLAT

        if (
            self.config.candidate_id
            == FROZEN_CONFIGS[MeanReversionCandidate.MRL1].candidate_id
        ):
            if (
                hmm_state_int == self.config.hmm_state
                and self.config.volatility_low
                <= volatility
                < self.config.volatility_high
                and zscore_value <= -abs(self.config.zscore_threshold)
            ):
                return StrategySignal.LONG

            return StrategySignal.FLAT

        return StrategySignal.FLAT

    def get_risk_stop_price(
        self,
        *,
        entry_price: float,
        signal: StrategySignal,
        market_data: Mapping[str, Any] | None = None,
    ) -> float | None:
        """
        Calculate the protective stop from the frozen strategy geometry.

        LONG:
            stop = entry - stop_points

        SHORT:
            stop = entry + stop_points
        """

        entry = float(entry_price)

        if entry <= 0:
            raise ValueError("entry_price must be positive.")

        if signal is StrategySignal.LONG:
            return entry - self.config.stop_points

        if signal is StrategySignal.SHORT:
            return entry + self.config.stop_points

        return None
