from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)

from .config import MeanReversionConfig, MRS2_CONFIG
from .signal import generate_signal


class MeanReversionStrategy(BaseStrategy):
    """
    Modular Mean Reversion strategy.

    The strategy represents one frozen research candidate.

    It is intentionally stateless with respect to trade management.
    Position state, execution, risk, TP/SL handling, and broker interaction
    belong to the surrounding architecture.

    Frozen candidates:
        MRS2
        MRL1
        MRL2
    """

    def __init__(
        self,
        config: MeanReversionConfig | None = None,
    ) -> None:
        self.config = config or MRS2_CONFIG

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def candidate_id(self) -> str:
        return self.config.candidate_id

    @property
    def side(self) -> str:
        return self.config.side

    @property
    def target_points(self) -> float:
        return self.config.target_points

    @property
    def stop_points(self) -> float:
        return self.config.stop_points

    @property
    def horizon_bars(self) -> int:
        return self.config.horizon_bars

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        """Generate the frozen Mean Reversion signal."""

        return generate_signal(
            market_data=market_data,
            config=self.config,
        )

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        """
        Evaluate the frozen Mean Reversion entry conditions.
        """

        signal = self.generate_signal(market_data)

        if signal is StrategySignal.LONG:
            return StrategyDecision(
                signal=StrategySignal.LONG,
                action=StrategyAction.ENTER,
                reason=(f"{self.name} entry conditions satisfied."),
            )

        if signal is StrategySignal.SHORT:
            return StrategyDecision(
                signal=StrategySignal.SHORT,
                action=StrategyAction.ENTER,
                reason=(f"{self.name} entry conditions satisfied."),
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason=(f"{self.name} entry conditions not satisfied."),
        )
