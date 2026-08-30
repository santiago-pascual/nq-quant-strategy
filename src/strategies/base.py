from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StrategySignal(Enum):
    """Directional output emitted by a strategy."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class StrategyAction(Enum):
    """Action requested by a strategy."""

    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"


@dataclass(frozen=True)
class StrategyDecision:
    """Deterministic decision emitted by a strategy."""

    signal: StrategySignal
    action: StrategyAction
    reason: str | None = None


class BaseStrategy(ABC):
    """
    Abstract interface for strategy components.

    A strategy transforms market information and its internal strategy state
    into a deterministic trading decision.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable strategy name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the strategy implementation version."""

    @abstractmethod
    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        """
        Generate a directional signal from current market information.
        """

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        """
        Evaluate the strategy and return a deterministic decision.
        """

        signal = self.generate_signal(market_data)

        if signal is StrategySignal.FLAT:
            action = StrategyAction.HOLD
        else:
            action = StrategyAction.ENTER

        return StrategyDecision(
            signal=signal,
            action=action,
        )

    def on_fill(
        self,
        *,
        market_data: Mapping[str, Any],
        position: Any,
    ) -> None:
        """
        Optional lifecycle hook called after an execution fill.

        Stateful strategies may override this method.
        """

    def on_market_data(
        self,
        market_data: Mapping[str, Any],
        position: Any,
    ) -> StrategyDecision | None:
        """
        Optional lifecycle hook for an active position.

        Stateful strategies may return an EXIT decision.
        """

        return None

    def on_exit(self) -> None:
        """
        Optional lifecycle hook called after a strategy exit is executed.
        """
