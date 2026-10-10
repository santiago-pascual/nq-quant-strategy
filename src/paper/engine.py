from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime
from zoneinfo import ZoneInfo

from typing import Any, Iterable

import numpy as np

from src.broker import (
    BrokerAdapter,
    BrokerFill,
    BrokerOrderStatus,
    InMemoryBrokerAdapter,
)

from src.execution import ExecutionEngine

from src.paper.lifecycle import PositionLifecycleController

from src.paper.logger import PaperEventLogger, PaperEventType

from src.paper.context_adapter import PaperMarketContextAdapter

from src.portfolio.broker_execution import BrokerExecutionCoordinator

from src.portfolio.conflict import EntryRequest, PortfolioConflictEngine

from src.risk import RiskEngine, RiskRequest

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)
from src.strategies.s2r.strategy import S2RStrategy


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
    initial_equity: float = 50_000.0
    tick_size: float = 0.25
    price_offset: float = 0.0
    commission_per_contract: float = 0.0
    exchange_fee_per_contract: float = 0.0
    regulatory_fee_per_contract: float = 0.0
    automatic_simulated_fills: bool = False

    def __post_init__(self) -> None:
        if self.point_value <= 0 or self.initial_equity <= 0 or self.tick_size <= 0:
            raise ValueError("point_value, initial_equity, and tick_size must be positive.")
        if (self.price_offset < 0 or self.commission_per_contract < 0
                or self.exchange_fee_per_contract < 0 or self.regulatory_fee_per_contract < 0):
            raise ValueError("price offset and per-contract costs must be non-negative.")


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
    account_equity: float = 0.0
    realized_pnl: float = 0.0
    commissions: float = 0.0
    exchange_fees: float = 0.0
    regulatory_fees: float = 0.0


# ============================================================================

# PAPER TRADING ENGINE

# ============================================================================


class PaperTradingEngine:
    """

    Deterministic paper-trading orchestration engine.

    Pipeline

    --------

        Raw Market Data

             ↓

        Paper Market Context

             ↓

        Causal Features + HMM

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

    - build causal market context when configured

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
        context_adapter: PaperMarketContextAdapter | None = None,
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

        self.context_adapter = context_adapter

        self.config = config or PaperEngineConfig()

        # Broker ↔ execution bridge.

        #

        # IMPORTANT:

        # BrokerExecutionCoordinator uses the real parameter name

        # \\\\\`execution_engine\\\\\`, not \\\\\`execution\\\\\`.

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

        #   partial fill -> position exists -> risk registration

        #   full fill    -> remaining pending authorization removed

        self._pending_risk_results: dict[str, Any] = {}

        # Entry requests are kept until the corresponding broker entry

        # is completely filled.

        self._pending_entry_requests: dict[str, EntryRequest] = {}
        self._automatic_simulated_fills = self.config.automatic_simulated_fills
        if self._automatic_simulated_fills and not isinstance(
            self.broker, InMemoryBrokerAdapter
        ):
            raise TypeError(
                "Automatic simulated fills require InMemoryBrokerAdapter."
            )
        self._simulated_orders: dict[
            str, tuple[str, datetime, float | None]
        ] = {}
        self._pending_strategy_orders: dict[str, set[str]] = {}
        self._newly_filled_strategies: set[str] = set()
        self._realized_pnl = 0.0
        self._commissions = 0.0
        self._exchange_fees = 0.0
        self._regulatory_fees = 0.0
        self._account_equity = self.config.initial_equity
        self._position_commissions: dict[str, float] = {}
        self._position_exchange_fees: dict[str, float] = {}
        self._position_regulatory_fees: dict[str, float] = {}
        self._position_initial_risk: dict[str, float] = {}

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

    @property
    def last_market_data(self) -> dict[str, Any] | None:
        return dict(self._last_market_data) if self._last_market_data is not None else None

    @property
    def account_equity(self) -> float:
        return self._account_equity

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def commissions(self) -> float:
        return self._commissions

    @property
    def exchange_fees(self) -> float:
        return self._exchange_fees

    @property
    def regulatory_fees(self) -> float:
        return self._regulatory_fees

    @property
    def total_costs(self) -> float:
        return self._commissions + self._exchange_fees + self._regulatory_fees

    def enable_simulated_fills(self) -> None:
        if not isinstance(self.broker, InMemoryBrokerAdapter):
            raise TypeError(
                "Automatic simulated fills require InMemoryBrokerAdapter."
            )
        self._automatic_simulated_fills = True

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

    # MARKET CONTEXT

    # ========================================================================

    def _build_market_context(
        self,
        market_data: dict[str, Any],
        *,
        context_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Enrich one raw OHLCV observation using the stateful causal
        market-context adapter.

        The causal path consumes exactly one completed bar at a time.
        ``context_index`` is retained only for compatibility with existing
        callers; it is intentionally ignored by the causal implementation.
        """

        if self.context_adapter is None:
            return dict(market_data)

        return self.context_adapter.update(market_data)

    def _configure_window_models(self, market_data: dict[str, Any]) -> None:
        scheduled_windows = market_data.get("s2r_windows")
        if scheduled_windows is not None:
            if self.context_adapter is None:
                raise RuntimeError("S2R Research windows require a model context adapter.")
            windows = tuple(int(value) for value in scheduled_windows)
            models = {
                window: self.context_adapter.context.s2_model_for_window(window)
                for window in windows
            }
            for strategy in self.strategies:
                if isinstance(strategy, S2RStrategy):
                    strategy.set_fitted_models_for_bar(models)
            return
        window = market_data.get("hmm_window")
        if self.context_adapter is None:
            return
        # Autonomous live S2R receives its own independently scheduled
        # two-year/three-month raw-state HMM and fitted training quantities.
        # Research Replay continues to use the explicit window mapping above.
        live_hash = market_data.get("s2r_model_hash")
        live_version = market_data.get("s2r_model_version")
        provider = getattr(self.context_adapter.context, "hmm_provider", None)
        live_model = getattr(provider, "current_s2r_fitted_model", None)
        if live_hash and live_version is not None and live_model is not None:
            for strategy in self.strategies:
                if isinstance(strategy, S2RStrategy) and strategy.model_window != int(live_version):
                    strategy.set_fitted_model(live_model, window=int(live_version))
            return
        if market_data.get("hmm_state") is None or window is None:
            return
        for strategy in self.strategies:
            if isinstance(strategy, S2RStrategy) and strategy.model_window != int(window):
                strategy.set_fitted_model(
                    self.context_adapter.context.s2_model_for_window(int(window)),
                    window=int(window),
                )

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
            if self._pending_strategy_orders.get(strategy.name):
                decision = StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.HOLD,
                    reason="broker order pending",
                )
                decisions[strategy.name] = decision
                self.logger.log_strategy_decision(
                    strategy_name=strategy.name,
                    action=decision.action.value,
                    signal=decision.signal.value,
                    timestamp=timestamp,
                    reason=decision.reason,
                )
                continue
            if (
                strategy.name in self._newly_filled_strategies
                and self.lifecycle.has_position(strategy.name)
            ):
                decision = StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.HOLD,
                    reason="entry bar is not managed",
                )
                decisions[strategy.name] = decision
                self.logger.log_strategy_decision(
                    strategy_name=strategy.name,
                    action=decision.action.value,
                    signal=decision.signal.value,
                    timestamp=timestamp,
                    reason=decision.reason,
                )
                continue
            lifecycle_result = self.lifecycle.evaluate(
                strategy_name=strategy.name,
                market_data=market_data,
                timestamp=timestamp,
            )

            decision = lifecycle_result.decision

            if isinstance(strategy, S2RStrategy):
                for evaluation in strategy.take_window_evaluations():
                    self.logger.append(
                        PaperEventType.CANDIDATE_EVALUATION,
                        {
                            "strategy_name": strategy.name,
                            "evaluation_id": (
                                f"S2R|{timestamp.isoformat()}|window-{evaluation['window']}"
                            ),
                            "entry_timestamp": timestamp.isoformat(),
                            "session_id": timestamp.astimezone(
                                ZoneInfo("America/New_York")
                            ).date().isoformat(),
                            **evaluation,
                        },
                        timestamp=timestamp,
                    )

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

        broker entry is filled.

        """

        if decision.action is not StrategyAction.ENTER:
            return None

        if decision.signal is StrategySignal.FLAT:
            raise ValueError(
                f"Strategy {strategy.name} produced ENTER with FLAT signal"
            )

        entry_price = float(
            strategy.get_entry_reference_price(
                signal=decision.signal,
                market_data=market_data,
            )
        )

        # The strategy owns its risk geometry. The paper engine remains

        # strategy-agnostic and asks the strategy for its protective stop.

        stop_price = strategy.get_risk_stop_price(
            entry_price=entry_price,
            signal=decision.signal,
            market_data=market_data,
        )

        if stop_price is None:
            stop_price = market_data.get("risk_stop_price")

        if stop_price is None:
            raise ValueError(
                f"Strategy {strategy.name} did not provide a risk stop price "
                "for an ENTER decision."
            )

        stop_price = float(stop_price)

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
        risk_per_contract = (
            abs(entry_price - stop_price) * self.config.point_value
        )
        risk_budget_override = None
        adaptive_quantity = None
        sizing_rule = {
            "ORB": "ORB_STRICT_INTEGER",
            "MRL1": "MR_STRICT_025",
            "MRS2": "MR_STRICT_025",
        }.get(strategy.name, "STRICT_RISK_BUDGET")
        strategy_risk_cap = strategy.single_contract_risk_cap
        if (
            strategy.name == "ORB"
            and strategy_risk_cap is not None
            and risk_per_contract > self.risk.limits.risk_per_trade
            and risk_per_contract <= strategy_risk_cap
        ):
            adaptive_quantity = 1
            sizing_rule = (
                "ORB_FORCE_1_UNDER_060"
                if strategy.name == "ORB"
                else "SINGLE_CONTRACT_RISK_CAP"
            )
        elif strategy.name == "ORB" and strategy_risk_cap is not None:
            if risk_per_contract > strategy_risk_cap:
                sizing_rule = "ORB_REJECT_OVER_060"

        self.logger.log_risk_request(
            {
                "strategy_name": strategy.name,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "point_value": self.config.point_value,
                # Persist a stable numeric representation across checkpoint
                # restore (which restores account balances as floats). This
                # affects event identity only; risk evaluation still uses the
                # unchanged RiskRequest/account value below.
                "account_equity": float(account_equity),
                "risk_per_contract": risk_per_contract,
                "risk_budget_override": risk_budget_override,
                "adaptive_quantity": adaptive_quantity,
                "sizing_rule": sizing_rule,
            },
            timestamp=timestamp,
        )

        risk_result = self.risk.evaluate(
            risk_request,
            trading_day=timestamp.astimezone(
                ZoneInfo("America/New_York")
            ).date(),
            risk_per_trade_budget=risk_budget_override,
            adaptive_quantity=adaptive_quantity,
            adaptive_risk_limit=(
                strategy_risk_cap if adaptive_quantity is not None else None
            ),
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
                "risk_budget_override": risk_budget_override,
                "theoretical_quantity": risk_result.theoretical_quantity,
                "executable_quantity": risk_result.executable_quantity,
                "adaptive_quantity": risk_result.adaptive_quantity,
                "sizing_rule": sizing_rule,
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
        self._register_pending_order(
            broker_order.broker_order_id,
            strategy.name,
            timestamp,
            strategy.get_entry_fill_price(
                signal=decision.signal,
                market_data=market_data,
            ),
        )

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
        market_data: dict[str, Any] | None = None,
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
        self._register_pending_order(
            broker_order.broker_order_id,
            strategy.name,
            timestamp,
            strategy.get_exit_fill_price(market_data=market_data or {}),
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

    def _register_pending_order(
        self,
        broker_order_id: str,
        strategy_name: str,
        timestamp: datetime,
        fill_price_override: float | None,
    ) -> None:
        self._pending_strategy_orders.setdefault(strategy_name, set()).add(
            broker_order_id
        )
        if self._automatic_simulated_fills:
            self._simulated_orders[broker_order_id] = (
                strategy_name,
                timestamp,
                fill_price_override,
            )

    def _advance_simulated_fills(
        self,
        market_data: dict[str, Any],
        timestamp: datetime,
        *,
        immediate_only: bool,
    ) -> list[BrokerFill]:
        if not self._automatic_simulated_fills:
            return []
        if not isinstance(self.broker, InMemoryBrokerAdapter):
            raise RuntimeError("Simulated fills are configured for a non-simulated broker.")

        results: list[BrokerFill] = []
        for broker_order_id, (
            strategy_name,
            submitted_at,
            override_price,
        ) in list(self._simulated_orders.items()):
            immediate = override_price is not None and submitted_at == timestamp
            due = immediate if immediate_only else (
                override_price is None and submitted_at < timestamp
            )
            if not due:
                continue
            order = self.broker.get_order(broker_order_id)
            if order is None or order.status not in {
                BrokerOrderStatus.SUBMITTED,
                BrokerOrderStatus.PARTIALLY_FILLED,
            }:
                self._simulated_orders.pop(broker_order_id, None)
                self._discard_pending_order(strategy_name, broker_order_id)
                continue
            if override_price is None:
                base_price = float(market_data["open"])
            else:
                base_price = float(override_price)
            offset = (
                np.ceil(self.config.price_offset / self.config.tick_size)
                * self.config.tick_size
            )
            direction = 1.0 if order.request.signal is StrategySignal.LONG else -1.0
            fill_price = base_price + direction * offset
            fill = self.broker.process_fill(
                broker_order_id,
                quantity=order.request.quantity,
                price=fill_price,
            )
            self.process_broker_fill(fill, timestamp=timestamp)
            results.append(fill)
        return results

    def _discard_pending_order(self, strategy_name: str, broker_order_id: str) -> None:
        pending = self._pending_strategy_orders.get(strategy_name)
        if pending is None:
            return
        pending.discard(broker_order_id)
        if not pending:
            self._pending_strategy_orders.pop(strategy_name, None)

    def _refresh_pending_orders(self) -> None:
        for strategy_name, order_ids in list(self._pending_strategy_orders.items()):
            for broker_order_id in tuple(order_ids):
                broker_order = self.broker.get_order(broker_order_id)
                if broker_order is None or broker_order.status not in {
                    BrokerOrderStatus.CANCELLED,
                    BrokerOrderStatus.REJECTED,
                }:
                    continue
                self._simulated_orders.pop(broker_order_id, None)
                self._pending_risk_results.pop(broker_order_id, None)
                self._pending_entry_requests.pop(broker_order_id, None)
                self._discard_pending_order(strategy_name, broker_order_id)

    # ========================================================================

    # MARKET BAR

    # ========================================================================

    def process_bar(
        self,
        market_data: dict[str, Any],
        *,
        account_equity: float | None = None,
        context_index: int | None = None,
    ) -> PaperStepResult:
        """

        Process one completed market-data observation.

        If a causal context adapter is configured, the raw market-data

        observation is enriched one completed bar at a time before strategy/lifecycle evaluation.

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

        # Build causal market context

        # ------------------------------------------------------------------

        market_data = self._build_market_context(
            market_data,
            context_index=context_index,
        )
        self._configure_window_models(market_data)

        # The context engine is authoritative for the enriched timestamp.

        timestamp = market_data["timestamp"]

        if not isinstance(timestamp, datetime):
            raise TypeError("enriched market_data timestamp must be datetime")
        if account_equity is not None:
            if account_equity <= 0:
                raise ValueError("account_equity must be positive.")
            if self._last_timestamp is None:
                self._account_equity = float(account_equity)

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

        trading_day = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        if (
            self._last_timestamp is None
            or trading_day
            != self._last_timestamp.astimezone(ZoneInfo("America/New_York")).date()
        ):
            self.risk.reset_day(
                trading_day,
            )

        self._last_market_data = dict(market_data)
        self._newly_filled_strategies.clear()
        fills = self._advance_simulated_fills(
            market_data,
            timestamp,
            immediate_only=False,
        )
        self._refresh_pending_orders()

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
                        market_data,
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
                    self._account_equity,
                )

                if broker_order is not None:
                    submitted_orders.append(
                        broker_order,
                    )

        fills.extend(
            self._advance_simulated_fills(
                market_data,
                timestamp,
                immediate_only=True,
            )
        )

        # ------------------------------------------------------------------

        # Runtime state

        # ------------------------------------------------------------------

        self._last_timestamp = timestamp

        self._last_decisions = dict(decisions)
        self._newly_filled_strategies.clear()

        positions = tuple(self.execution.get_positions())

        return PaperStepResult(
            timestamp=timestamp,
            decisions=decisions,
            submitted_orders=tuple(submitted_orders),
            fills=tuple(fills),
            open_positions=positions,
            account_equity=self._account_equity,
            realized_pnl=self._realized_pnl,
            commissions=self._commissions,
            exchange_fees=self._exchange_fees,
            regulatory_fees=self._regulatory_fees,
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
        commission = broker_fill.quantity * self.config.commission_per_contract
        exchange_fee = broker_fill.quantity * self.config.exchange_fee_per_contract
        regulatory_fee = broker_fill.quantity * self.config.regulatory_fee_per_contract
        total_fill_cost = commission + exchange_fee + regulatory_fee
        self._commissions += commission
        self._exchange_fees += exchange_fee
        self._regulatory_fees += regulatory_fee
        self._account_equity -= total_fill_cost
        self._position_commissions[strategy_name] = (
            self._position_commissions.get(strategy_name, 0.0) + commission
        )
        self._position_exchange_fees[strategy_name] = (
            self._position_exchange_fees.get(strategy_name, 0.0) + exchange_fee
        )
        self._position_regulatory_fees[strategy_name] = (
            self._position_regulatory_fees.get(strategy_name, 0.0) + regulatory_fee
        )

        broker_order = self.broker.get_order(broker_fill.broker_order_id)
        if broker_order is not None and self._is_filled_status(broker_order.status):
            self._simulated_orders.pop(broker_fill.broker_order_id, None)
            self._discard_pending_order(strategy_name, broker_fill.broker_order_id)

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
                "commission": commission,
                "exchange_fee": exchange_fee,
                "regulatory_fee": regulatory_fee,
                "total_cost": total_fill_cost,
                "artificial_slippage_ticks": 0,
                "contract_symbol": (self._last_market_data or {}).get("contract_symbol"),
            },
            timestamp=timestamp,
        )

        # ------------------------------------------------------------------

        # ENTRY FILL

        # ------------------------------------------------------------------

        if execution_action is StrategyAction.ENTER:
            if position_after is None:
                raise RuntimeError("Entry fill did not create an execution position")
            self._newly_filled_strategies.add(strategy_name)

            # First fill.

            if position_before is None:
                self.logger.log_position_opened(
                    {
                        "strategy_name": strategy_name,
                        "quantity": position_after.quantity,
                        "entry_price": position_after.entry_price,
                        "side": position_after.side.value,
                        "contract_symbol": (self._last_market_data or {}).get("contract_symbol"),
                        "entry_features": self._serializable_market_data(self._last_market_data or {}),
                        "point_value": self.config.point_value,
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

            per_contract_risk = getattr(risk_result, "risk_per_contract", None)
            if per_contract_risk is not None:
                self._position_initial_risk[strategy_name] = (
                    self._position_initial_risk.get(strategy_name, 0.0)
                    + float(per_contract_risk) * broker_fill.quantity
                )

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

            broker_order = self.broker.get_order(broker_fill.broker_order_id)

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
            self._realized_pnl += realized_pnl
            self._account_equity += realized_pnl
            total_position_cost = (
                self._position_commissions[strategy_name]
                + self._position_exchange_fees.get(strategy_name, 0.0)
                + self._position_regulatory_fees.get(strategy_name, 0.0)
            )
            initial_risk_usd = self._position_initial_risk.get(strategy_name)
            net_pnl = realized_pnl - total_position_cost
            self.logger.log_position_closed(
                {
                    "strategy_name": strategy_name,
                    "quantity": position_before.quantity,
                    "entry_price": position_before.entry_price,
                    "side": position_before.side.value,
                    "exit_price": broker_fill.price,
                    "point_value": self.config.point_value,
                    "gross_pnl": realized_pnl,
                    "commission": self._position_commissions[strategy_name],
                    "exchange_fees": self._position_exchange_fees.get(strategy_name, 0.0),
                    "regulatory_fees": self._position_regulatory_fees.get(strategy_name, 0.0),
                    "total_costs": total_position_cost,
                    "net_pnl": net_pnl,
                    "initial_risk_usd": initial_risk_usd,
                    "realized_r": net_pnl / initial_risk_usd if initial_risk_usd else None,
                    "account_balance_after": self._account_equity,
                },
                timestamp=timestamp,
            )

            # Risk release follows the actual quantity that was exited.

            self.risk.register_exit_fill(
                strategy_name,
                fill_quantity=position_before.quantity,
                realized_pnl=(
                    net_pnl
                ),
            )
            self._position_commissions.pop(strategy_name, None)
            self._position_exchange_fees.pop(strategy_name, None)
            self._position_regulatory_fees.pop(strategy_name, None)
            self._position_initial_risk.pop(strategy_name, None)

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

    @staticmethod
    def _is_filled_status(status: Any) -> bool:
        return (
            status is BrokerOrderStatus.FILLED
            or getattr(status, "value", None) == BrokerOrderStatus.FILLED.value
            or getattr(status, "name", None) == BrokerOrderStatus.FILLED.name
        )
