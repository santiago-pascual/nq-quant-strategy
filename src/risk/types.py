from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk limits used by the live/paper risk engine.

    Values are expressed in account currency unless noted otherwise.
    """

    risk_per_trade: float
    max_total_risk: float
    max_daily_loss: float
    max_concurrent_positions: int
    max_daily_trades: int
    max_contracts: int

    def __post_init__(self) -> None:
        if self.risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be positive")
        if self.max_total_risk <= 0:
            raise ValueError("max_total_risk must be positive")
        if self.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if self.max_daily_trades <= 0:
            raise ValueError("max_daily_trades must be positive")
        if self.max_contracts <= 0:
            raise ValueError("max_contracts must be positive")


@dataclass(frozen=True)
class RiskRequest:
    """Request to size and authorize one new position."""

    strategy_name: str
    entry_price: float
    stop_price: float
    point_value: float
    account_equity: float

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.stop_price <= 0:
            raise ValueError("stop_price must be positive")
        if self.point_value <= 0:
            raise ValueError("point_value must be positive")
        if self.account_equity <= 0:
            raise ValueError("account_equity must be positive")

        if self.entry_price == self.stop_price:
            raise ValueError("entry_price and stop_price must differ")


@dataclass(frozen=True)
class RiskResult:
    decision: RiskDecision
    strategy_name: str
    quantity: int
    risk_per_contract: float
    total_risk: float
    reason: str

    @property
    def approved(self) -> bool:
        return self.decision is RiskDecision.APPROVED
