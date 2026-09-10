from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)

from .config import (
    MRS2_CONFIG,
    MeanReversionConfig,
)
from .context import MeanReversionContextBuilder


class MeanReversionStrategy(BaseStrategy):
    """
    Modular Mean Reversion strategy.

    One instance represents one frozen research candidate.

    The strategy consumes precomputed market context:
        - HMM state
        - volatility percentile
        - z-score

    It does not perform fitting, data loading, or research optimization.
    """

    def __init__(
        self,
        config: MeanReversionConfig | None = None,
    ) -> None:
        """
        Initialize the strategy.

        If no configuration is supplied, MRS2 is used as the
        default frozen candidate so the BaseStrategy contract
        can instantiate the strategy without arguments.
        """

        self.config = config or MRS2_CONFIG
        self._context_builder = MeanReversionContextBuilder()

    @property
    def name(self) -> str:
        """Return the frozen candidate name."""
        return self.config.name

    @property
    def version(self) -> str:
        """Return the strategy implementation version."""
        return "1.0.0"

    def build_context(
        self,
        market_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Build the market context required by Mean Reversion.
        """

        return self._context_builder.build(market_data)

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        """
        Generate the frozen Mean Reversion entry signal.

        LONG:
            zscore <= -zscore_threshold

        SHORT:
            zscore >= +zscore_threshold
        """

        context = self.build_context(market_data)

        hmm_state = context.get("hmm_state")
        volatility_percentile = context.get("vol_percentile")
        zscore = context.get("zscore")

        # Required context must be present.
        if hmm_state is None:
            return StrategySignal.FLAT

        if volatility_percentile is None:
            return StrategySignal.FLAT

        if zscore is None:
            return StrategySignal.FLAT

        # Reject boolean values explicitly.
        # bool is a subclass of int in Python.
        if isinstance(hmm_state, bool):
            return StrategySignal.FLAT

        # HMM regime filter.
        if int(hmm_state) != self.config.hmm_state:
            return StrategySignal.FLAT

        # Volatility percentile filter.
        #
        # Lower boundary is inclusive.
        # Upper boundary is exclusive.
        if not (
            self.config.volatility_low
            <= float(volatility_percentile)
            < self.config.volatility_high
        ):
            return StrategySignal.FLAT

        threshold = float(self.config.zscore_threshold)
        current_zscore = float(zscore)

        # Directional z-score condition.
        if self.config.side == "LONG":
            if current_zscore <= -threshold:
                return StrategySignal.LONG

        elif self.config.side == "SHORT":
            if current_zscore >= threshold:
                return StrategySignal.SHORT

        else:
            raise ValueError(f"Unknown Mean Reversion side: {self.config.side!r}")

        return StrategySignal.FLAT

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        """
        Evaluate the frozen Mean Reversion candidate.
        """

        signal = self.generate_signal(market_data)

        if signal is StrategySignal.LONG:
            return StrategyDecision(
                signal=StrategySignal.LONG,
                action=StrategyAction.ENTER,
                reason=(f"{self.config.name} long entry conditions satisfied."),
            )

        if signal is StrategySignal.SHORT:
            return StrategyDecision(
                signal=StrategySignal.SHORT,
                action=StrategyAction.ENTER,
                reason=(f"{self.config.name} short entry conditions satisfied."),
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason=(f"{self.config.name} entry conditions not satisfied."),
        )
