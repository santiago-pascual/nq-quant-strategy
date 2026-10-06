from __future__ import annotations

from dataclasses import dataclass

from .types import RiskLimits


@dataclass(frozen=True)
class ProductionRiskPolicy:
    """
    Production risk policy for the normal $50K Topstep XFA.

    This policy is intentionally separate from the test fixtures used
    throughout the risk/paper tests.

    Important:
        account_size is the nominal XFA size used to calculate the
        strategy risk budget.

        account_equity is NOT assumed to be $50,000. A Topstep XFA
        starts with a $0 trading balance and grows through P&L.
    """

    account_size: float = 50_000.0

    # Frozen from the full-system XFA robustness/funded simulation.
    risk_fraction: float = 0.0025

    # Internal portfolio controls.
    max_total_risk_fraction: float | None = 0.0075
    max_concurrent_positions: int = 3
    max_daily_trades: int = 10

    # Internal emergency loss guard.
    #
    # This is deliberately below the firm's $2,000 50K XFA MLL.
    # It is NOT the Topstep MLL itself.
    emergency_daily_loss: float = 500.0

    # Paper implementation ceiling.
    #
    # The actual Topstep XFA Scaling Plan is balance-dependent.
    # This value is an internal upper bound only; the dynamic scaling
    # layer will be added separately.
    max_contracts: int = 20

    def __post_init__(self) -> None:
        if self.account_size <= 0:
            raise ValueError("account_size must be positive")

        if self.risk_fraction <= 0:
            raise ValueError("risk_fraction must be positive")

        if (
            self.max_total_risk_fraction is not None
            and self.max_total_risk_fraction <= 0
        ):
            raise ValueError("max_total_risk_fraction must be positive")

        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")

        if self.max_daily_trades <= 0:
            raise ValueError("max_daily_trades must be positive")

        if self.emergency_daily_loss <= 0:
            raise ValueError("emergency_daily_loss must be positive")

        if self.max_contracts <= 0:
            raise ValueError("max_contracts must be positive")

        if (
            self.max_total_risk is not None
            and self.max_total_risk < self.risk_per_trade
        ):
            raise ValueError("max_total_risk must be >= risk_per_trade")

    @property
    def risk_per_trade(self) -> float:
        """
        Dollar risk budget for one trade.

        $50,000 * 0.25% = $125.
        """
        return self.account_size * self.risk_fraction

    @property
    def max_total_risk(self) -> float | None:
        """
        Maximum aggregate planned risk across simultaneously open
        positions.

        50,000 * 0.75% = $375.
        """
        if self.max_total_risk_fraction is None:
            return None
        return self.account_size * self.max_total_risk_fraction

    def to_risk_limits(self) -> RiskLimits:
        """
        Convert the frozen production policy into the existing
        RiskEngine configuration.
        """
        return RiskLimits(
            risk_per_trade=self.risk_per_trade,
            max_total_risk=self.max_total_risk,
            max_daily_loss=self.emergency_daily_loss,
            max_concurrent_positions=self.max_concurrent_positions,
            max_daily_trades=self.max_daily_trades,
            max_contracts=self.max_contracts,
        )


XFA_50K_PRODUCTION_POLICY = ProductionRiskPolicy()


@dataclass(frozen=True)
class ResearchReplayRiskPolicy(ProductionRiskPolicy):
    """Research-equivalent sizing without the production aggregate-risk cap."""

    max_total_risk_fraction: float | None = None


RESEARCH_REPLAY_RISK_POLICY = ResearchReplayRiskPolicy()
