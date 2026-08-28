from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any


class StrategySignal(Enum):
    """Directional output emitted by a strategy for the current observation."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class BaseStrategy(ABC):
    """Abstract interface for pure signal-generation strategy components.

    The strategy layer is responsible for transforming available market
    information into a directional trading signal.

    The strategy layer is not responsible for account rules, position sizing,
    execution, commissions, slippage, broker integration, or regime-routing
    policy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable strategy name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the strategy implementation version identifier."""

    @abstractmethod
    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        """Generate a LONG, SHORT, or FLAT signal from current market data.

        Implementations should use only information available at the current
        decision point and must not embed execution or account-management
        behavior in this method.
        """
