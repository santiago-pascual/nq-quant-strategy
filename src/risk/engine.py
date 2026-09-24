from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .types import RiskDecision, RiskLimits, RiskRequest, RiskResult


@dataclass
class _OpenPosition:
    strategy_name: str
    risk_per_contract: float
    quantity: int
    total_risk: float


@dataclass
class RiskEngine:
    """Deterministic pre-trade and execution-aware risk engine.

    The engine has two distinct responsibilities:

    1. Pre-trade:
       - validate a proposed position
       - calculate authorized quantity
       - enforce account-level limits

    2. Execution bookkeeping:
       - track actual filled exposure
       - support partial entry fills
       - support partial exit fills
       - release risk only as exposure is actually reduced

    The engine does not generate signals, calculate strategy stops,
    or select production risk-policy values.
    """

    limits: RiskLimits
    _open_positions: dict[str, _OpenPosition] = field(default_factory=dict)
    _daily_trade_count: int = 0
    _daily_realized_pnl: float = 0.0
    _current_day: date | None = None

    def reset_day(self, trading_day: date) -> None:
        if self._current_day != trading_day:
            self._current_day = trading_day
            self._daily_trade_count = 0
            self._daily_realized_pnl = 0.0

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def daily_trade_count(self) -> int:
        return self._daily_trade_count

    @property
    def daily_realized_pnl(self) -> float:
        return self._daily_realized_pnl

    @property
    def open_risk(self) -> float:
        return sum(position.total_risk for position in self._open_positions.values())

    def evaluate(
        self,
        request: RiskRequest,
        *,
        trading_day: date,
    ) -> RiskResult:
        """Size and authorize a proposed position.

        Quantity is determined exclusively from the configured risk limits
        and proposed entry/stop distance.

        Actual open exposure is registered later through execution fills.
        """

        self.reset_day(trading_day)

        if self._daily_realized_pnl <= -self.limits.max_daily_loss:
            return self._reject(
                request,
                "daily loss limit reached",
            )

        if request.strategy_name in self._open_positions:
            return self._reject(
                request,
                "strategy already has an open position",
            )

        if self.open_position_count >= self.limits.max_concurrent_positions:
            return self._reject(
                request,
                "maximum concurrent positions reached",
            )

        if self._daily_trade_count >= self.limits.max_daily_trades:
            return self._reject(
                request,
                "maximum daily trades reached",
            )

        risk_per_contract = (
            abs(request.entry_price - request.stop_price) * request.point_value
        )

        quantity = int(self.limits.risk_per_trade // risk_per_contract)

        if quantity <= 0:
            return self._reject(
                request,
                "stop distance is too large for the configured risk per trade",
                risk_per_contract=risk_per_contract,
            )

        quantity = min(quantity, self.limits.max_contracts)
        total_risk = quantity * risk_per_contract

        if self.open_risk + total_risk > self.limits.max_total_risk:
            return self._reject(
                request,
                "maximum aggregate open risk would be exceeded",
                quantity=quantity,
                risk_per_contract=risk_per_contract,
                total_risk=total_risk,
            )

        return RiskResult(
            decision=RiskDecision.APPROVED,
            strategy_name=request.strategy_name,
            quantity=quantity,
            risk_per_contract=risk_per_contract,
            total_risk=total_risk,
            reason="risk checks passed",
        )

    def register_entry_fill(
        self,
        result: RiskResult,
        *,
        fill_quantity: int,
    ) -> None:
        """Register actual filled entry exposure.

        This method is intentionally fill-based rather than order-based.

        Example:
            authorized = 20
            fill 1    -> risk exposure = 1 contract
            fill 2    -> risk exposure = 2 contracts
            ...
            fill 20   -> risk exposure = 20 contracts

        The daily trade count is incremented only when the first actual
        entry fill creates the position.
        """

        if not result.approved:
            raise ValueError("cannot register a rejected risk result")

        if fill_quantity <= 0:
            raise ValueError("fill_quantity must be positive")

        position = self._open_positions.get(result.strategy_name)

        if position is None:
            if self.open_position_count >= self.limits.max_concurrent_positions:
                raise RuntimeError("maximum concurrent positions reached")

            if (
                self.open_risk + (fill_quantity * result.risk_per_contract)
                > self.limits.max_total_risk
            ):
                raise RuntimeError("maximum aggregate open risk reached")

            if fill_quantity > result.quantity:
                raise ValueError("entry fill exceeds risk-authorized quantity")

            total_risk = fill_quantity * result.risk_per_contract

            self._open_positions[result.strategy_name] = _OpenPosition(
                strategy_name=result.strategy_name,
                risk_per_contract=result.risk_per_contract,
                quantity=fill_quantity,
                total_risk=total_risk,
            )

            self._daily_trade_count += 1
            return

        if position.risk_per_contract != result.risk_per_contract:
            raise RuntimeError("risk per contract changed during partial entry")

        new_quantity = position.quantity + fill_quantity

        if new_quantity > result.quantity:
            raise ValueError("cumulative entry fills exceed risk-authorized quantity")

        additional_risk = fill_quantity * result.risk_per_contract

        if self.open_risk + additional_risk > self.limits.max_total_risk:
            raise RuntimeError("maximum aggregate open risk reached")

        position.quantity = new_quantity
        position.total_risk += additional_risk

    def register_exit_fill(
        self,
        strategy_name: str,
        *,
        fill_quantity: int,
        realized_pnl: float = 0.0,
    ) -> None:
        """Reduce actual open exposure by an executed exit fill.

        The realized P&L is accumulated only when the position becomes flat.
        """

        if fill_quantity <= 0:
            raise ValueError("fill_quantity must be positive")

        position = self._open_positions.get(strategy_name)

        if position is None:
            raise RuntimeError("strategy has no open position")

        if fill_quantity > position.quantity:
            raise ValueError("exit fill exceeds current risk position quantity")

        position.quantity -= fill_quantity
        position.total_risk = position.quantity * position.risk_per_contract

        if position.quantity == 0:
            del self._open_positions[strategy_name]
            self._daily_realized_pnl += realized_pnl

    def register_entry(self, result: RiskResult) -> None:
        """Backward-compatible full-fill registration.

        Existing callers that operate in an all-or-nothing manner can still
        register a completed position through this method.
        """

        self.register_entry_fill(
            result,
            fill_quantity=result.quantity,
        )

    def register_exit(
        self,
        strategy_name: str,
        realized_pnl: float,
    ) -> None:
        """Backward-compatible full-position exit registration."""

        position = self._open_positions.get(strategy_name)

        if position is None:
            raise RuntimeError("strategy has no open position")

        self.register_exit_fill(
            strategy_name,
            fill_quantity=position.quantity,
            realized_pnl=realized_pnl,
        )

    def _reject(
        self,
        request: RiskRequest,
        reason: str,
        *,
        quantity: int = 0,
        risk_per_contract: float = 0.0,
        total_risk: float = 0.0,
    ) -> RiskResult:
        return RiskResult(
            decision=RiskDecision.REJECTED,
            strategy_name=request.strategy_name,
            quantity=quantity,
            risk_per_contract=risk_per_contract,
            total_risk=total_risk,
            reason=reason,
        )
