from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.broker import BrokerAdapter, BrokerFill
from src.execution import ExecutionEngine
from src.paper.logger import PaperEventLogger
from src.portfolio.broker_execution import (
    BrokerExecutionCoordinator,
)
from src.portfolio.conflict import (
    EntryRequest,
    PortfolioConflictEngine,
)
from src.risk.engine import RiskEngine
from src.risk.types import RiskRequest
from src.strategies.base import (
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


@dataclass(frozen=True)
class DryRunResult:
    strategy_name: str
    decision: StrategyDecision
    risk_approved: bool
    risk_quantity: int
    broker_order_id: str | None
    fill_id: str | None
    position_open: bool
    event_count: int


class DryRunEngine:
    """
    End-to-end dry-run orchestration.

    Flow:

        Strategy
            ↓
        Risk
            ↓
        Portfolio Conflict
            ↓
        Execution
            ↓
        Broker
            ↓
        Fill
            ↓
        Position
            ↓
        Paper Logger

    Production risk-policy values are intentionally not defined here.
    """

    def __init__(
        self,
        *,
        execution: ExecutionEngine,
        risk: RiskEngine,
        conflict: PortfolioConflictEngine,
        broker: BrokerAdapter,
        logger: PaperEventLogger,
    ) -> None:

        self.execution = execution
        self.risk = risk
        self.conflict = conflict

        self.broker_execution = BrokerExecutionCoordinator(
            execution_engine=execution,
            broker=broker,
        )

        self.logger = logger

    # ------------------------------------------------------------------
    # CONNECTION
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.broker_execution.connect()

    def disconnect(self) -> None:
        self.broker_execution.disconnect()

    # ------------------------------------------------------------------
    # POSITION LOOKUP
    # ------------------------------------------------------------------

    def _find_position(
        self,
        strategy_name: str,
    ):
        positions = self.execution.get_positions()

        return next(
            (
                position
                for position in positions
                if position.strategy_name == strategy_name
            ),
            None,
        )

    # ------------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------------

    def run_entry(
        self,
        *,
        decision: StrategyDecision,
        strategy_name: str,
        strategy_version: str,
        timestamp: datetime,
        entry_price: float,
        stop_price: float,
        point_value: float,
        account_equity: float,
        market_data: dict[str, Any] | None = None,
    ) -> DryRunResult:

        if market_data is not None:
            self.logger.log_market_data(
                market_data,
                timestamp=timestamp,
            )

        # --------------------------------------------------------------
        # Strategy decision.
        # --------------------------------------------------------------

        self.logger.log_strategy_decision(
            strategy_name=strategy_name,
            action=decision.action.value,
            signal=decision.signal.value,
            timestamp=timestamp,
            reason=decision.reason,
        )

        # --------------------------------------------------------------
        # No entry.
        # --------------------------------------------------------------

        if decision.action is not StrategyAction.ENTER:
            return DryRunResult(
                strategy_name=strategy_name,
                decision=decision,
                risk_approved=False,
                risk_quantity=0,
                broker_order_id=None,
                fill_id=None,
                position_open=False,
                event_count=self.logger.count(),
            )

        if decision.signal not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            return DryRunResult(
                strategy_name=strategy_name,
                decision=decision,
                risk_approved=False,
                risk_quantity=0,
                broker_order_id=None,
                fill_id=None,
                position_open=False,
                event_count=self.logger.count(),
            )

        # --------------------------------------------------------------
        # Risk request.
        # --------------------------------------------------------------

        risk_request = RiskRequest(
            strategy_name=strategy_name,
            entry_price=entry_price,
            stop_price=stop_price,
            point_value=point_value,
            account_equity=account_equity,
        )

        self.logger.log_risk_request(
            {
                "strategy_name": strategy_name,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "point_value": point_value,
                "account_equity": account_equity,
            },
            timestamp=timestamp,
        )

        risk_result = self.risk.evaluate(
            risk_request,
            trading_day=timestamp.date(),
        )

        self.logger.log_risk_decision(
            {
                "strategy_name": strategy_name,
                "decision": risk_result.decision.value,
                "approved": risk_result.approved,
                "quantity": risk_result.quantity,
                "risk_per_contract": (risk_result.risk_per_contract),
                "total_risk": risk_result.total_risk,
                "reason": risk_result.reason,
            },
            timestamp=timestamp,
        )

        # --------------------------------------------------------------
        # Risk rejection.
        # --------------------------------------------------------------

        if not risk_result.approved:
            return DryRunResult(
                strategy_name=strategy_name,
                decision=decision,
                risk_approved=False,
                risk_quantity=0,
                broker_order_id=None,
                fill_id=None,
                position_open=False,
                event_count=self.logger.count(),
            )

        # --------------------------------------------------------------
        # Portfolio conflict.
        # --------------------------------------------------------------

        conflict_request = EntryRequest(
            strategy_name=strategy_name,
            side=decision.signal.value,
            quantity=risk_result.quantity,
        )

        conflict_result = self.conflict.evaluate(
            conflict_request,
        )

        if not conflict_result.approved:
            self.logger.log_error(
                {
                    "stage": "portfolio_conflict",
                    "strategy_name": strategy_name,
                    "reason": conflict_result.reason,
                },
                timestamp=timestamp,
            )

            return DryRunResult(
                strategy_name=strategy_name,
                decision=decision,
                risk_approved=False,
                risk_quantity=0,
                broker_order_id=None,
                fill_id=None,
                position_open=False,
                event_count=self.logger.count(),
            )

        # --------------------------------------------------------------
        # Register approved portfolio entry.
        # --------------------------------------------------------------

        self.conflict.register_entry(
            conflict_request,
        )

        # --------------------------------------------------------------
        # Build execution intent.
        # --------------------------------------------------------------

        execution_intent = self.execution.build_intent(
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            decision=decision,
            timestamp=timestamp,
        )

        # --------------------------------------------------------------
        # Submit to broker.
        # --------------------------------------------------------------

        broker_order = self.broker_execution.submit_entry(
            execution_intent,
            quantity=risk_result.quantity,
        )

        submission = self.broker_execution.submission(
            broker_order.broker_order_id,
        )

        if submission is None:
            raise RuntimeError("Broker submission disappeared after entry submission")

        self.logger.log_order_submitted(
            {
                "broker_order_id": (broker_order.broker_order_id),
                "execution_order_id": (submission.execution_order_id),
                "strategy_name": strategy_name,
                "strategy_version": strategy_version,
                "signal": decision.signal.value,
                "quantity": risk_result.quantity,
            },
            timestamp=timestamp,
        )

        return DryRunResult(
            strategy_name=strategy_name,
            decision=decision,
            risk_approved=True,
            risk_quantity=risk_result.quantity,
            broker_order_id=broker_order.broker_order_id,
            fill_id=None,
            position_open=False,
            event_count=self.logger.count(),
        )

    # ------------------------------------------------------------------
    # BROKER FILL
    # ------------------------------------------------------------------

    def process_fill(
        self,
        *,
        broker_fill: BrokerFill,
        timestamp: datetime,
    ) -> DryRunResult:

        # --------------------------------------------------------------
        # Resolve broker submission.
        # --------------------------------------------------------------

        submission = self.broker_execution.broker_order(
            broker_fill.broker_order_id,
        )

        if submission is None:
            raise KeyError("Unknown broker order for dry-run fill")

        broker_order = submission.broker_order

        strategy_name = broker_order.request.strategy_name

        position_before = self._find_position(
            strategy_name,
        )

        # --------------------------------------------------------------
        # Log broker fill.
        # --------------------------------------------------------------

        self.logger.log_fill(
            {
                "broker_order_id": (broker_fill.broker_order_id),
                "fill_id": broker_fill.fill_id,
                "strategy_name": strategy_name,
                "quantity": broker_fill.quantity,
                "price": broker_fill.price,
                "signal": broker_fill.signal.value,
            },
            timestamp=timestamp,
        )

        # --------------------------------------------------------------
        # Broker → Execution.
        # --------------------------------------------------------------

        execution_fill = self.broker_execution.process_broker_fill(
            broker_fill,
            timestamp=timestamp,
        )

        position_after = self._find_position(
            strategy_name,
        )

        # --------------------------------------------------------------
        # Position lifecycle.
        # --------------------------------------------------------------

        if position_before is None and position_after is not None:
            self.logger.log_position_opened(
                {
                    "strategy_name": strategy_name,
                    "quantity": position_after.quantity,
                    "entry_price": position_after.entry_price,
                    "signal": position_after.side.value,
                },
                timestamp=timestamp,
            )

        elif position_before is not None and position_after is not None:
            self.logger.log_position_updated(
                {
                    "strategy_name": strategy_name,
                    "quantity": position_after.quantity,
                    "entry_price": position_after.entry_price,
                    "signal": position_after.side.value,
                },
                timestamp=timestamp,
            )

        elif position_before is not None and position_after is None:
            self.logger.log_position_closed(
                {
                    "strategy_name": strategy_name,
                    "quantity": position_before.quantity,
                    "entry_price": position_before.entry_price,
                    "signal": position_before.side.value,
                },
                timestamp=timestamp,
            )

        # --------------------------------------------------------------
        # Return.
        # --------------------------------------------------------------

        return DryRunResult(
            strategy_name=strategy_name,
            decision=StrategyDecision(
                signal=submission.execution_intent.signal,
                action=submission.execution_intent.action,
                reason=submission.execution_intent.reason,
            ),
            risk_approved=True,
            risk_quantity=broker_fill.quantity,
            broker_order_id=broker_fill.broker_order_id,
            fill_id=execution_fill.fill_id,
            position_open=position_after is not None,
            event_count=self.logger.count(),
        )
