from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)

from .config import ORBConfig
from .context import ORBContext
from .signal import generate_orb_signal


class ORBStrategy(BaseStrategy):
    """
    Modular Opening Range Breakout strategy.
    """

    def __init__(
        self,
        config: ORBConfig | None = None,
    ) -> None:
        self.config = config or ORBConfig()
        self._context: ORBContext | None = None
        self._in_trade = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def in_trade(self) -> bool:
        return self._in_trade

    def set_context(self, context: ORBContext) -> None:
        self._context = context

    def reset(self) -> None:
        self._context = None
        self._in_trade = False

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        if self._in_trade:
            return StrategySignal.FLAT

        if self._context is None:
            raise RuntimeError(
                "ORB context must be set before generating a signal."
            )

        return generate_orb_signal(
            market_data,
            self._context,
        )

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        signal = self.generate_signal(market_data)

        if signal is StrategySignal.LONG:
            return StrategyDecision(
                signal=signal,
                action=StrategyAction.ENTER,
                reason="ORB upper breakout.",
            )

        if signal is StrategySignal.SHORT:
            return StrategyDecision(
                signal=signal,
                action=StrategyAction.ENTER,
                reason="ORB lower breakout.",
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason="No ORB breakout.",
        )

    def start_trade(self) -> None:
        if self._in_trade:
            raise RuntimeError(
                "Cannot start ORB trade while another trade is active."
            )

        self._in_trade = True

    def finish_trade(self) -> None:
        self._in_trade = False
