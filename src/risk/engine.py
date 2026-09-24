from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .types import RiskDecision, RiskLimits, RiskRequest, RiskResult


@dataclass
class _OpenPosition:
    strategy_name: str
    risk: float


@dataclass
class RiskEngine:
    """Deterministic pre-trade risk engine.

    This layer does not generate signals, submit orders, manage fills,
    or calculate strategy stops. It only validates a proposed trade,
    sizes it, and tracks aggregate risk/account limits.
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
        return sum(position.risk for position in self._open_positions.values())

    def evaluate(self, request: RiskRequest, *, trading_day: date) -> RiskResult:
        """Size and authorize a proposed position.

        Quantity is determined exclusively from the frozen risk limits
        and the proposed entry/stop distance:

            risk_per_contract =
                abs(entry - stop) * point_value

            quantity =
                floor(risk_per_trade / risk_per_contract)

        No strategy performance data is used here.
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
            abs(request.entry_price - request.stop_price)
            * request.point_value
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

    def register_entry(self, result: RiskResult) -> None:
        """Register an approved position after execution confirms a fill."""
        if not result.approved:
            raise ValueError("cannot register a rejected risk result")

        if result.strategy_name in self._open_positions:
            raise RuntimeError("strategy already has an open position")

        if self.open_position_count >= self.limits.max_concurrent_positions:
            raise RuntimeError("maximum concurrent positions reached")

        if self.open_risk + result.total_risk > self.limits.max_total_risk:
            raise RuntimeError("maximum aggregate open risk reached")

        self._open_positions[result.strategy_name] = _OpenPosition(
            strategy_name=result.strategy_name,
            risk=result.total_risk,
        )
        self._daily_trade_count += 1

    def register_exit(self, strategy_name: str, realized_pnl: float) -> None:
        """Remove an open position and record realized daily P&L."""
        if strategy_name not in self._open_positions:
            raise RuntimeError("strategy has no open position")

        del self._open_positions[strategy_name]
        self._daily_realized_pnl += realized_pnl

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
