from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.domain import Order, Position
from src.strategies.base import StrategyDecision


@dataclass(frozen=True)
class RiskDecision:
    """
    Immutable result of a risk evaluation.

    approved:
        Whether the proposed action is allowed.

    order:
        Risk-adjusted order to be sent to the execution layer.
        None when the action is rejected or no order is required.

    reason:
        Human-readable explanation for audit and debugging.
    """

    approved: bool
    order: Order | None = None
    reason: str | None = None


class RiskManager(ABC):
    """
    Abstract risk-management interface.

    The risk layer is responsible for deciding whether a strategy decision
    may be executed and, when appropriate, determining the resulting order.

    The risk layer is NOT responsible for:

        - generating strategy signals
        - market regime classification
        - broker communication
        - order execution
        - strategy-specific entry logic
        - strategy-specific exit logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the risk manager name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the risk implementation version."""

    @abstractmethod
    def evaluate(
        self,
        decision: StrategyDecision,
        position: Position | None,
    ) -> RiskDecision:
        """
        Evaluate a strategy decision against current portfolio state.

        Implementations may apply position sizing, account limits,
        exposure constraints, drawdown rules, or kill-switch logic.
        """

    @abstractmethod
    def validate_order(self, order: Order) -> RiskDecision:
        """
        Validate an already constructed order against risk constraints.

        This provides a second safety boundary before execution.
        """
