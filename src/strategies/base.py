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

    The strategy layer is NOT responsible for:

    - account management
    - position sizing
    - risk limits
    - order execution
    - broker integration
    - commissions
    - slippage
    - portfolio management
    - regime selection
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

        Implementations must use only information available at the current
        decision point and must not use future information.
        """

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        """
        Evaluate the strategy and return a deterministic decision.

        The default implementation maps the directional signal to an action.
        Stateful strategies may override this method when their decision
        depends on an existing position or internal strategy state.
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
