from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from src.broker import BrokerAdapter, BrokerFill
from src.execution import ExecutionEngine
from src.paper.lifecycle import PositionLifecycleController
from src.paper.logger import PaperEventLogger
from src.portfolio.broker_execution import BrokerExecutionCoordinator
from src.portfolio.conflict import EntryRequest, PortfolioConflictEngine
from src.risk import RiskEngine, RiskRequest
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class PaperEngineConfig:
    """
    Configuration for the paper-trading orchestration layer.

    This deliberately contains no production risk policy.
    Risk limits belong to RiskEngine / the production risk-policy stage.
    """

    point_value: float = 2.0


# ============================================================================
# STEP RESULT
# ============================================================================


@dataclass(frozen=True)
class PaperStepResult:
    """
    Result of processing one market-data observation.
    """

    timestamp: datetime
    decisions: dict[str, StrategyDecision]
    submitted_orders: tuple[Any, ...]
    fills: tuple[Any, ...]
    open_positions: tuple[Any, ...]


# ============================================================================
# PAPER TRADING ENGINE
# ============================================================================


class PaperTradingEngine:
    """
    Deterministic paper-trading orchestration engine.

    Pipeline
    --------

        Market Data
             ↓
        Strategy / Lifecycle
             ↓
        Risk
             ↓
        Portfolio Conflict
             ↓
        Execution
             ↓
        Broker Adapter
             ↓
        Broker Fill
             ↓
        Execution Position
             ↓
        Strategy Lifecycle Hooks
             ↓
        Paper Logger

    Responsibilities
    ----------------
    - orchestrate the production components
    - evaluate strategies
    - route active-position decisions through lifecycle control
    - authorize entries through risk
    - enforce portfolio conflicts
    - submit orders through broker execution
    - process broker fills
    - notify strategy lifecycle hooks
    - log state transitions

    Explicitly NOT responsible for
    ------------------------------
    - fitting
    - optimization
    - strategy research
    - parameter changes
    - TP/SL implementation
    - production risk-policy selection
    - broker-specific logic
    """

    def __init__(
        self,
        *,
        strategies: Iterable[BaseStrategy],
        execution: ExecutionEngine,
        risk: RiskEngine,
        conflict: PortfolioConflictEngine,
        broker: BrokerAdapter,
        logger: PaperEventLogger,
        config: PaperEngineConfig | None = None,
    ) -> None:
        self.strategies = tuple(strategies)

        if not self.strategies:
            raise ValueError("At least one strategy is required")

        strategy_names = [strategy.name for strategy in self.strategies]

        if len(strategy_names) != len(set(strategy_names)):
            raise ValueError("PaperTradingEngine requires unique strategy names")

        self.execution = execution
        self.risk = risk
        self.conflict = conflict
        self.broker = broker
        self.logger = logger
        self.config = config or PaperEngineConfig()

        # Broker ↔ execution bridge.
        #
        # IMPORTANT:
        # BrokerExecutionCoordinator uses the real parameter name
        # `execution_engine`, not `execution`.
        self.broker_execution = BrokerExecutionCoordinator(
            execution_engine=self.execution,
            broker=self.broker,
        )

        # Position lifecycle controller.
        self.lifecycle = PositionLifecycleController(
            execution=self.execution,
            strategies=self.strategies,
        )

        # Runtime state.
        self._running = False
        self._last_timestamp: datetime | None = None
        self._last_market_data: dict[str, Any] | None = None
        self._last_decisions: dict[str, StrategyDecision] = {}

        # Risk results are kept until the corresponding broker entry
        # is completely filled.
        #
        # This is essential for partial-fill correctness:
        #
        #   partial fill -> position exists -> NO risk registration
        #   full fill    -> risk registration
        self._pending_risk_results: dict[str, Any] = {}

        # Entry requests are kept until the corresponding broker entry
        # is completely filled.
        self._pending_entry_requests: dict[str, EntryRequest] = {}

    # ========================================================================
    # CONNECTION LIFECYCLE
    # ========================================================================

    @property
    def running(self) -> bool:
        return self._running

    def connect(self) -> None:
        """
        Connect the broker execution layer and start the engine.
        """

        if self._running:
            return

        self.broker_execution.connect()
        self._running = True

    def stop(self) -> None:
        """
        Stop processing new market data.

        Existing state is intentionally preserved.
        """

        self._running = False

    def disconnect(self) -> None:
        """
        Stop the engine and disconnect the broker layer.
        """

        self._running = False
        self.broker_execution.disconnect()

    # ========================================================================
    # PUBLIC STATE ACCESS
    # ========================================================================

    @property
    def last_timestamp(self) -> datetime | None:
        return self._last_timestamp

    @property
    def last_decisions(self) -> dict[str, StrategyDecision]:
        return dict(self._last_decisions)

    def position(self, strategy_name: str):
        return self.execution.get_position(strategy_name)

    def has_position(self, strategy_name: str) -> bool:
        return self.lifecycle.has_position(strategy_name)

    # ========================================================================
    # INTERNAL LOOKUPS
    # ========================================================================

    def _strategy(self, strategy_name: str) -> BaseStrategy:
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                return strategy

        raise KeyError(f"Unknown strategy: {strategy_name}")

    # ========================================================================
    # TIMESTAMP VALIDATION
    # ========================================================================

    def _validate_timestamp(
        self,
        timestamp: datetime,
    ) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("market_data timestamp must be timezone-aware")

        if timestamp.utcoffset() is None:
            raise ValueError("market_data timestamp must be timezone-aware")

        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError(
                    "market_data timestamps must be strictly chronological"
                )

    # ========================================================================
    # MARKET DATA SERIALIZATION
    # ========================================================================

    @staticmethod
    def _serializable_market_data(
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert runtime market data into JSON-safe values.

        PaperEventLogger serializes payloads with json.dumps(), so datetime
        objects inside the payload must be converted to ISO strings.
        """

        serialized: dict[str, Any] = {}

        for key, value in market_data.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value

        return serialized

    # ========================================================================
    # STRATEGY EVALUATION
    # ========================================================================

    def _evaluate_strategies(
        self,
        market_data: dict[str, Any],
        timestamp: datetime,
    ) -> dict[str, StrategyDecision]:
        """
        Evaluate every strategy through PositionLifecycleController.

        Flat strategy:
            lifecycle controller returns HOLD/no_open_position.

        Active position:
            lifecycle controller delegates to strategy.on_market_data().
        """

        decisions: dict[str, StrategyDecision] = {}

        for strategy in self.strategies:
            lifecycle_result = self.lifecycle.evaluate(
                strategy_name=strategy.name,
                market_data=market_data,
                timestamp=timestamp,
            )

            decision = lifecycle_result.decision

            decisions[strategy.name] = decision

            self.logger.log_strategy_decision(
                strategy_name=strategy.name,
                action=decision.action.value,
                signal=decision.signal.value,
                timestamp=timestamp,
                reason=decision.reason,
            )

        return decisions

    # ========================================================================
    # ENTRY
    # ========================================================================

    def _try_entry(
        self,
        strategy: BaseStrategy,
        decision: StrategyDecision,
        market_data: dict[str, Any],
        timestamp: datetime,
        account_equity: float,
    ):
        """
        Execute the entry pipeline:

            Strategy
                ↓
            Risk
                ↓
            Portfolio Conflict
                ↓
            Execution
                ↓
            Broker

        Portfolio/risk state is committed only after the corresponding
        broker entry is completely filled.
        """

        if decision.action is not StrategyAction.ENTER:
            return None

        if decision.signal is StrategySignal.FLAT:
            raise ValueError(
                f"Strategy {strategy.name} produced ENTER with FLAT signal"
            )

        entry_price = float(market_data["close"])

        stop_price = float(
            market_data.get(
                "risk_stop_price",
                entry_price,
            )
        )

        # ------------------------------------------------------------------
        # Risk request
        # ------------------------------------------------------------------

        risk_request = RiskRequest(
            strategy_name=strategy.name,
            entry_price=entry_price,
            stop_price=stop_price,
            point_value=self.config.point_value,
            account_equity=account_equity,
        )

        self.logger.log_risk_request(
            {
                "strategy_name": strategy.name,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "point_value": self.config.point_value,
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
                "strategy_name": strategy.name,
                "decision": risk_result.decision.value,
                "approved": risk_result.approved,
                "quantity": risk_result.quantity,
                "risk_per_contract": risk_result.risk_per_contract,
                "total_risk": risk_result.total_risk,
                "reason": risk_result.reason,
            },
            timestamp=timestamp,
        )

        if not risk_result.approved:
            return None

        quantity = int(risk_result.quantity)

        if quantity <= 0:
            raise RuntimeError(
                f"Risk engine approved {strategy.name} with non-positive quantity"
            )

        # ------------------------------------------------------------------
        # Portfolio conflict
        # ------------------------------------------------------------------

        entry_request = EntryRequest(
            strategy_name=strategy.name,
            side=decision.signal.value,
            quantity=quantity,
        )

        conflict_result = self.conflict.evaluate(
            entry_request,
        )

        if not conflict_result.approved:
            self.logger.log_error(
                f"Portfolio conflict rejected entry: {conflict_result.reason}",
                timestamp=timestamp,
                context={
                    "strategy_name": strategy.name,
                },
            )
            return None

        # ------------------------------------------------------------------
        # Execution intent
        # ------------------------------------------------------------------

        intent = self.execution.build_intent(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            decision=decision,
            timestamp=timestamp,
        )

        # ------------------------------------------------------------------
        # Broker submission
        # ------------------------------------------------------------------

        broker_order = self.broker_execution.submit_entry(
            intent,
            quantity=quantity,
        )

        # Keep risk/conflict information pending until the broker order
        # reaches a complete fill.
        self._pending_risk_results[broker_order.broker_order_id] = risk_result

        self._pending_entry_requests[broker_order.broker_order_id] = entry_request

        self.logger.log_order_created(
            {
                "strategy_name": strategy.name,
                "strategy_version": strategy.version,
                "action": decision.action.value,
                "signal": decision.signal.value,
                "quantity": quantity,
                "broker_order_id": broker_order.broker_order_id,
                "reason": decision.reason,
            },
            timestamp=timestamp,
        )

        self.logger.log_order_submitted(
            {
                "strategy_name": strategy.name,
                "strategy_version": strategy.version,
                "quantity": quantity,
                "broker_order_id": broker_order.broker_order_id,
            },
            timestamp=timestamp,
        )

        return broker_order

    # ========================================================================
    # EXIT
    # ========================================================================

    def _try_exit(
        self,
        strategy: BaseStrategy,
        decision: StrategyDecision,
        timestamp: datetime,
    ):
        """
        Submit an EXIT generated by an active strategy.

        EXIT must always use FLAT as its strategy signal.
        """

        if decision.action is not StrategyAction.EXIT:
            return None

        if decision.signal is not StrategySignal.FLAT:
            raise ValueError(
                f"Strategy {strategy.name} produced EXIT without FLAT signal"
            )

        position = self.lifecycle.position(
            strategy.name,
        )

        if position is None:
            raise RuntimeError(
                f"Strategy {strategy.name} requested EXIT without an open position"
            )

        intent = self.execution.build_intent(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            decision=decision,
            timestamp=timestamp,
        )

        broker_order = self.broker_execution.submit_exit(
            intent,
        )

        self.logger.log_order_created(
            {
                "strategy_name": strategy.name,
                "strategy_version": strategy.version,
                "action": decision.action.value,
                "signal": decision.signal.value,
                "quantity": position.quantity,
                "broker_order_id": broker_order.broker_order_id,
                "reason": decision.reason,
            },
            timestamp=timestamp,
        )

        self.logger.log_order_submitted(
            {
                "strategy_name": strategy.name,
                "strategy_version": strategy.version,
                "quantity": position.quantity,
                "broker_order_id": broker_order.broker_order_id,
            },
            timestamp=timestamp,
        )

        return broker_order

    # ========================================================================
    # MARKET BAR
    # ========================================================================

    def process_bar(
        self,
        market_data: dict[str, Any],
        *,
        account_equity: float,
    ) -> PaperStepResult:
        """
        Process one completed market-data observation.
        """

        if not self._running:
            raise RuntimeError("Paper trading engine is not running")

        if "timestamp" not in market_data:
            raise ValueError("market_data must contain timestamp")

        timestamp = market_data["timestamp"]

        if not isinstance(timestamp, datetime):
            raise TypeError("market_data timestamp must be datetime")

        self._validate_timestamp(timestamp)

        # ------------------------------------------------------------------
        # Market data log
        # ------------------------------------------------------------------

        self.logger.log_market_data(
            self._serializable_market_data(market_data),
            timestamp=timestamp,
        )

        # ------------------------------------------------------------------
        # Risk trading-day reset
        # ------------------------------------------------------------------

        if (
            self._last_timestamp is None
            or timestamp.date() != self._last_timestamp.date()
        ):
            self.risk.reset_day(
                timestamp.date(),
            )

        # ------------------------------------------------------------------
        # Strategy/lifecycle evaluation
        # ------------------------------------------------------------------

        decisions = self._evaluate_strategies(
            market_data,
            timestamp,
        )

        submitted_orders: list[Any] = []

        # ------------------------------------------------------------------
        # Route decisions
        # ------------------------------------------------------------------

        for strategy in self.strategies:
            decision = decisions[strategy.name]

            if self.lifecycle.has_position(
                strategy.name,
            ):
                if decision.action is StrategyAction.EXIT:
                    broker_order = self._try_exit(
                        strategy,
                        decision,
                        timestamp,
                    )

                    if broker_order is not None:
                        submitted_orders.append(
                            broker_order,
                        )

                continue

            if decision.action is StrategyAction.ENTER:
                broker_order = self._try_entry(
                    strategy,
                    decision,
                    market_data,
                    timestamp,
                    account_equity,
                )

                if broker_order is not None:
                    submitted_orders.append(
                        broker_order,
                    )

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------

        self._last_timestamp = timestamp
        self._last_market_data = dict(market_data)
        self._last_decisions = dict(decisions)

        positions = tuple(self.execution.get_positions())

        return PaperStepResult(
            timestamp=timestamp,
            decisions=decisions,
            submitted_orders=tuple(submitted_orders),
            fills=(),
            open_positions=positions,
        )

    # ========================================================================
    # BROKER FILL
    # ========================================================================

    def process_broker_fill(
        self,
        broker_fill: BrokerFill,
        *,
        timestamp: datetime,
    ):
        """
        Process a broker fill through:

            Broker
              ↓
            Execution
              ↓
            Position
              ↓
            Strategy lifecycle
              ↓
            Risk / conflict bookkeeping
        """

        if timestamp.tzinfo is None:
            raise ValueError("broker fill timestamp must be timezone-aware")

        if timestamp.utcoffset() is None:
            raise ValueError("broker fill timestamp must be timezone-aware")

        # ------------------------------------------------------------------
        # Resolve coordinated submission
        # ------------------------------------------------------------------

        submission = self.broker_execution.submission(
            broker_fill.broker_order_id,
        )

        if submission is None:
            raise KeyError("Unknown broker order for paper fill")

        strategy_name = submission.execution_intent.strategy_name

        execution_action = submission.execution_intent.action

        position_before = self.lifecycle.position(
            strategy_name,
        )

        # ------------------------------------------------------------------
        # Broker → Execution
        # ------------------------------------------------------------------

        execution_fill = self.broker_execution.process_broker_fill(
            broker_fill,
            timestamp=timestamp,
        )

        position_after = self.lifecycle.position(
            strategy_name,
        )

        # ------------------------------------------------------------------
        # Fill log
        # ------------------------------------------------------------------

        self.logger.log_fill(
            {
                "broker_order_id": broker_fill.broker_order_id,
                "fill_id": broker_fill.fill_id,
                "strategy_name": strategy_name,
                "quantity": broker_fill.quantity,
                "price": broker_fill.price,
                "signal": broker_fill.signal.value,
            },
            timestamp=timestamp,
        )

        # ------------------------------------------------------------------
        # ENTRY FILL
        # ------------------------------------------------------------------

        if execution_action is StrategyAction.ENTER:
            if position_after is None:
                raise RuntimeError("Entry fill did not create an execution position")

            # First fill.
            if position_before is None:
                self.logger.log_position_opened(
                    {
                        "strategy_name": strategy_name,
                        "quantity": position_after.quantity,
                        "entry_price": position_after.entry_price,
                        "side": position_after.side.value,
                    },
                    timestamp=timestamp,
                )

            # Additional/partial fill.
            else:
                self.logger.log_position_updated(
                    {
                        "strategy_name": strategy_name,
                        "quantity": position_after.quantity,
                        "entry_price": position_after.entry_price,
                        "side": position_after.side.value,
                    },
                    timestamp=timestamp,
                )

            # Notify strategy after every entry fill.
            if self._last_market_data is not None:
                self.lifecycle.notify_fill(
                    strategy_name=strategy_name,
                    market_data=self._last_market_data,
                )

                # --------------------------------------------------------------
            # Risk / portfolio bookkeeping follows ACTUAL fills.
            #
            # This is intentionally not delayed until BrokerOrderStatus.FILLED.
            # A partial fill creates real execution exposure and therefore
            # must create corresponding risk/conflict exposure immediately.
            # --------------------------------------------------------------

            risk_result = self._pending_risk_results.get(
                broker_fill.broker_order_id,
            )

            entry_request = self._pending_entry_requests.get(
                broker_fill.broker_order_id,
            )

            if risk_result is None:
                raise RuntimeError("Missing pending risk result for entry fill")

            if entry_request is None:
                raise RuntimeError("Missing pending portfolio entry for entry fill")

            # Register exactly the quantity that was actually filled.
            self.risk.register_entry_fill(
                risk_result,
                fill_quantity=broker_fill.quantity,
            )

            # Portfolio conflict is binary at strategy-position level:
            # the first actual fill occupies the strategy slot.
            if position_before is None:
                self.conflict.register_entry(
                    entry_request,
                )

            # Once the broker order is completely filled, no pending
            # authorization state is needed anymore.
            broker_order = self.broker.get_order(
                broker_fill.broker_order_id,
            )

            if broker_order is None:
                raise RuntimeError("Broker order disappeared after fill")

            from src.broker.types import BrokerOrderStatus

            if broker_order.status is BrokerOrderStatus.FILLED:
                self._pending_risk_results.pop(
                    broker_fill.broker_order_id,
                    None,
                )

                self._pending_entry_requests.pop(
                    broker_fill.broker_order_id,
                    None,
                )

            return execution_fill

        # ------------------------------------------------------------------
        # EXIT FILL
        # ------------------------------------------------------------------

        if execution_action is StrategyAction.EXIT:
            # Partial exit: position remains open.
            if position_after is not None:
                self.logger.log_position_updated(
                    {
                        "strategy_name": strategy_name,
                        "quantity": position_after.quantity,
                        "entry_price": position_after.entry_price,
                        "side": position_after.side.value,
                    },
                    timestamp=timestamp,
                )

                return execution_fill

            # Full exit.
            if position_before is None:
                raise RuntimeError("Exit fill closed no known execution position")

            self.logger.log_position_closed(
                {
                    "strategy_name": strategy_name,
                    "quantity": position_before.quantity,
                    "entry_price": position_before.entry_price,
                    "side": position_before.side.value,
                    "exit_price": broker_fill.price,
                },
                timestamp=timestamp,
            )

            # Realized PnL.
            if position_before.side is StrategySignal.LONG:
                price_difference = broker_fill.price - position_before.entry_price
            elif position_before.side is StrategySignal.SHORT:
                price_difference = position_before.entry_price - broker_fill.price
            else:
                raise RuntimeError("Cannot calculate PnL for FLAT position")

            realized_pnl = (
                price_difference * position_before.quantity * self.config.point_value
            )

            # Risk release follows the actual quantity that was exited.
            self.risk.register_exit_fill(
                strategy_name,
                fill_quantity=position_before.quantity,
                realized_pnl=realized_pnl,
            )

            # The execution position is now flat, so the portfolio slot
            # can be released.
            self.conflict.register_exit(
                strategy_name,
            )

            # Strategy lifecycle reset.
            self.lifecycle.notify_exit(
                strategy_name=strategy_name,
            )

            return execution_fill

        raise RuntimeError("Unsupported execution action for broker fill")
