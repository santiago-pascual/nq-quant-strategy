from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from src.execution import ExecutionEngine
from src.execution.types import Position
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


@dataclass(frozen=True)
class LifecycleDecision:
    strategy_name: str
    decision: StrategyDecision
    has_position: bool
    timestamp: datetime


class PositionLifecycleController:
    """
    Coordinates strategy evaluation according to position state.

    Lifecycle:

    1. No position:
       -> strategy.evaluate()
       -> normal entry evaluation.

    2. Open position:
       -> strategy.on_market_data()
       -> lifecycle/exit evaluation.

    3. Closed/non-open position:
       -> treated as flat
       -> strategy.evaluate().
    """

    def __init__(
        self,
        *,
        execution: ExecutionEngine,
        strategies: tuple[BaseStrategy, ...],
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")

        names = [strategy.name for strategy in strategies]

        if len(names) != len(set(names)):
            raise ValueError("Strategy names must be unique")

        self.execution = execution
        self._strategies = {strategy.name: strategy for strategy in strategies}

    # ------------------------------------------------------------------
    # Position access
    # ------------------------------------------------------------------

    def position(self, strategy_name: str) -> Position | None:
        return self.execution.get_position(strategy_name)

    def has_position(self, strategy_name: str) -> bool:
        position = self.position(strategy_name)

        return position is not None and position.status.value == "open"

    # ------------------------------------------------------------------
    # Strategy access
    # ------------------------------------------------------------------

    def _strategy(self, strategy_name: str) -> BaseStrategy:
        strategy = self._strategies.get(strategy_name)

        if strategy is None:
            raise KeyError(f"Unknown strategy: {strategy_name}")

        return strategy

    # ------------------------------------------------------------------
    # Main lifecycle evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        strategy_name: str,
        market_data: Mapping[str, Any],
        timestamp: datetime,
    ) -> LifecycleDecision:
        """
        Evaluate a strategy according to its current position state.
        """

        if timestamp.tzinfo is None:
            raise ValueError("Lifecycle timestamp must be timezone-aware")

        strategy = self._strategy(strategy_name)
        position = self.position(strategy_name)

        # ==============================================================
        # NO POSITION
        # ==============================================================

        if position is None:
            decision = strategy.evaluate(market_data)

            if decision is None:
                decision = StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.HOLD,
                    reason="no_open_position",
                )

            elif (
                decision.action is StrategyAction.HOLD
                and decision.signal is StrategySignal.FLAT
                and decision.reason is None
            ):
                decision = StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.HOLD,
                    reason="no_open_position",
                )

            self._validate_flat_strategy_decision(
                decision=decision,
            )

            return LifecycleDecision(
                strategy_name=strategy_name,
                decision=decision,
                has_position=False,
                timestamp=timestamp,
            )

        # ==============================================================
        # POSITION EXISTS BUT IS NOT OPEN
        # ==============================================================

        if position.status.value != "open":
            decision = strategy.evaluate(market_data)

            if decision is None:
                decision = StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.HOLD,
                    reason="position_not_open",
                )

            elif (
                decision.action is StrategyAction.HOLD
                and decision.signal is StrategySignal.FLAT
                and decision.reason is None
            ):
                decision = StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.HOLD,
                    reason="position_not_open",
                )

            self._validate_flat_strategy_decision(
                decision=decision,
            )

            return LifecycleDecision(
                strategy_name=strategy_name,
                decision=decision,
                has_position=False,
                timestamp=timestamp,
            )

        # ==============================================================
        # ACTIVE POSITION
        # ==============================================================

        decision = strategy.on_market_data(
            market_data,
            position,
        )

        if decision is None:
            decision = StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.HOLD,
                reason="strategy_lifecycle_hold",
            )

        self._validate_active_position_decision(
            decision=decision,
            position=position,
        )

        return LifecycleDecision(
            strategy_name=strategy_name,
            decision=decision,
            has_position=True,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Validation: flat state
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_flat_strategy_decision(
        *,
        decision: StrategyDecision,
    ) -> None:
        """
        Validate a decision emitted while no position is open.

        ENTER:
            must use LONG or SHORT.

        HOLD:
            must use FLAT.

        EXIT:
            invalid without an open position.
        """

        if decision.action is StrategyAction.ENTER:
            if decision.signal not in (
                StrategySignal.LONG,
                StrategySignal.SHORT,
            ):
                raise ValueError("ENTER decision must use LONG or SHORT signal")

        elif decision.action is StrategyAction.EXIT:
            raise ValueError("Strategy cannot emit EXIT while no position is open")

        elif decision.action is StrategyAction.HOLD:
            if decision.signal is not StrategySignal.FLAT:
                raise ValueError(
                    "HOLD decision without a position must use FLAT signal"
                )

    # ------------------------------------------------------------------
    # Validation: active position
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_active_position_decision(
        *,
        decision: StrategyDecision,
        position: Position,
    ) -> None:
        """
        Validate a decision emitted while a position is open.
        """

        if decision.action is StrategyAction.ENTER:
            raise ValueError("Strategy cannot emit ENTER while position is open")

        if decision.action is StrategyAction.EXIT:
            if decision.signal is not StrategySignal.FLAT:
                raise ValueError("EXIT decision must use FLAT signal")

        elif decision.action is StrategyAction.HOLD:
            if decision.signal not in (
                StrategySignal.FLAT,
                position.side,
            ):
                raise ValueError("HOLD decision signal is incompatible with position")

    # ------------------------------------------------------------------
    # Lifecycle notifications
    # ------------------------------------------------------------------

    def notify_fill(
        self,
        *,
        strategy_name: str,
        market_data: Mapping[str, Any],
    ) -> None:
        strategy = self._strategy(strategy_name)
        position = self.position(strategy_name)

        strategy.on_fill(
            market_data=market_data,
            position=position,
        )

    def notify_exit(
        self,
        *,
        strategy_name: str,
    ) -> None:
        strategy = self._strategy(strategy_name)
        strategy.on_exit()
