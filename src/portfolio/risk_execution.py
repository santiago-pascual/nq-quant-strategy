from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from src.execution.engine import ExecutionEngine
from src.execution.types import Order
from src.risk.engine import RiskEngine
from src.risk.types import RiskRequest, RiskResult
from src.strategies.base import StrategyDecision, StrategyAction


@dataclass(frozen=True)
class EntryApproval:
    """Complete result of the risk -> execution pre-trade gate."""

    risk_result: RiskResult
    order: Order | None

    @property
    def approved(self) -> bool:
        return self.risk_result.approved and self.order is not None


class RiskExecutionCoordinator:
    """Coordinate pre-trade risk authorization with order execution.

    This layer does not generate signals, calculate strategy stops/targets,
    allocate the portfolio, or communicate directly with a broker.
    """

    def __init__(
        self,
        *,
        risk_engine: RiskEngine,
        execution_engine: ExecutionEngine,
    ) -> None:
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine

    def authorize_entry(
        self,
        *,
        strategy_name: str,
        strategy_version: str,
        decision: StrategyDecision,
        timestamp: datetime,
        entry_price: float,
        stop_price: float,
        point_value: float,
        account_equity: float,
        trading_day: date,
    ) -> EntryApproval:
        """Run risk checks and submit an order only when approved."""

        if decision.action is not StrategyAction.ENTER:
            raise ValueError("authorize_entry requires an ENTER decision")

        if decision.signal.value == "flat":
            raise ValueError("authorize_entry requires a non-FLAT signal")

        request = RiskRequest(
            strategy_name=strategy_name,
            entry_price=entry_price,
            stop_price=stop_price,
            point_value=point_value,
            account_equity=account_equity,
        )

        risk_result = self.risk_engine.evaluate(
            request,
            trading_day=trading_day,
        )

        if not risk_result.approved:
            return EntryApproval(
                risk_result=risk_result,
                order=None,
            )

        intent = self.execution_engine.build_intent(
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            decision=decision,
            timestamp=timestamp,
        )

        order = self.execution_engine.submit_entry(
            intent,
            quantity=risk_result.quantity,
        )

        return EntryApproval(
            risk_result=risk_result,
            order=order,
        )

    def confirm_entry_fill(
        self,
        *,
        order_id: str,
        fill_price: float,
        timestamp: datetime,
    ):
        """Process an execution fill.

        Risk registration is intentionally separate because a partial fill
        must not be registered as a fully established position.
        """
        return self.execution_engine.process_entry_fill(
            order_id=order_id,
            fill_price=fill_price,
            timestamp=timestamp,
        )

    def register_approved_entry(
        self,
        risk_result: RiskResult,
    ) -> None:
        """Commit an approved risk reservation after confirmed full fill."""
        self.risk_engine.register_entry(risk_result)

    def register_exit(
        self,
        *,
        strategy_name: str,
        realized_pnl: float,
    ) -> None:
        """Release the strategy's open risk after a confirmed exit."""
        self.risk_engine.register_exit(
            strategy_name,
            realized_pnl,
        )
