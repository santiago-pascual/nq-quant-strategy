from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.lifecycle import PositionLifecycleController
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


TIMESTAMP = datetime(
    2026,
    1,
    2,
    15,
    0,
    tzinfo=timezone.utc,
)


class StubStrategy(BaseStrategy):
    name = "STUB"
    version = "1.0"

    def __init__(self) -> None:
        self.generate_calls = 0
        self.market_data_calls = 0
        self.fill_calls = 0
        self.exit_calls = 0

    def generate_signal(self, market_data):
        self.generate_calls += 1
        return StrategySignal.LONG

    def on_market_data(self, market_data, position):
        self.market_data_calls += 1

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.EXIT,
            reason="test_exit",
        )

    def on_fill(self, market_data, position):
        self.fill_calls += 1

    def on_exit(self):
        self.exit_calls += 1


class HoldStrategy(BaseStrategy):
    name = "HOLD"
    version = "1.0"

    def generate_signal(self, market_data):
        return StrategySignal.FLAT


class InvalidEntryStrategy(BaseStrategy):
    name = "INVALID_ENTRY"
    version = "1.0"

    def generate_signal(self, market_data):
        return StrategySignal.FLAT

    def evaluate(self, market_data):
        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.ENTER,
            reason="invalid",
        )


class InvalidExitFlatStrategy(BaseStrategy):
    name = "INVALID_EXIT"
    version = "1.0"

    def generate_signal(self, market_data):
        return StrategySignal.FLAT

    def evaluate(self, market_data):
        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.EXIT,
            reason="invalid",
        )


class InvalidActiveEntryStrategy(BaseStrategy):
    name = "INVALID_ACTIVE_ENTRY"
    version = "1.0"

    def generate_signal(self, market_data):
        return StrategySignal.LONG

    def on_market_data(self, market_data, position):
        return StrategyDecision(
            signal=position.side,
            action=StrategyAction.ENTER,
            reason="invalid",
        )


def make_execution() -> ExecutionEngine:
    return ExecutionEngine()


def make_market_data() -> dict:
    return {
        "timestamp": TIMESTAMP,
        "symbol": "MNQ.v.0",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 100,
    }


def make_entry_intent(
    execution: ExecutionEngine,
    *,
    strategy_name: str,
    strategy_version: str = "1.0",
    signal: StrategySignal = StrategySignal.LONG,
):
    decision = StrategyDecision(
        signal=signal,
        action=StrategyAction.ENTER,
        reason="lifecycle test entry",
    )

    return execution.build_intent(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        decision=decision,
        timestamp=TIMESTAMP,
    )


def make_exit_intent(
    execution: ExecutionEngine,
    *,
    strategy_name: str,
    strategy_version: str = "1.0",
):
    decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="lifecycle test exit",
    )

    return execution.build_intent(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        decision=decision,
        timestamp=TIMESTAMP,
    )


def test_controller_requires_strategy():
    execution = make_execution()

    with pytest.raises(ValueError, match="At least one strategy"):
        PositionLifecycleController(
            execution=execution,
            strategies=(),
        )


def test_controller_rejects_duplicate_strategy_names():
    execution = make_execution()

    first = StubStrategy()
    second = StubStrategy()

    with pytest.raises(ValueError, match="unique"):
        PositionLifecycleController(
            execution=execution,
            strategies=(first, second),
        )


def test_unknown_strategy_raises_key_error():
    execution = make_execution()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(StubStrategy(),),
    )

    with pytest.raises(KeyError, match="Unknown strategy"):
        controller.evaluate(
            strategy_name="UNKNOWN",
            market_data=make_market_data(),
            timestamp=TIMESTAMP,
        )


def test_timestamp_must_be_timezone_aware():
    execution = make_execution()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(StubStrategy(),),
    )

    naive_timestamp = datetime(2026, 1, 2, 15, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        controller.evaluate(
            strategy_name="STUB",
            market_data=make_market_data(),
            timestamp=naive_timestamp,
        )


def test_flat_strategy_is_evaluated_for_entry():
    execution = make_execution()
    strategy = StubStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    result = controller.evaluate(
        strategy_name="STUB",
        market_data=make_market_data(),
        timestamp=TIMESTAMP,
    )

    assert result.strategy_name == "STUB"
    assert result.has_position is False
    assert result.decision.signal is StrategySignal.LONG
    assert result.decision.action is StrategyAction.ENTER
    assert strategy.generate_calls == 1
    assert strategy.market_data_calls == 0


def test_hold_strategy_without_position_returns_hold():
    execution = make_execution()
    strategy = HoldStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    result = controller.evaluate(
        strategy_name="HOLD",
        market_data=make_market_data(),
        timestamp=TIMESTAMP,
    )

    assert result.has_position is False
    assert result.decision.signal is StrategySignal.FLAT
    assert result.decision.action is StrategyAction.HOLD


def test_active_position_uses_lifecycle_controller():
    execution = make_execution()
    strategy = StubStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    intent = make_entry_intent(
        execution,
        strategy_name="STUB",
        signal=StrategySignal.LONG,
    )

    order = execution.submit_entry(
        intent,
        quantity=1,
    )

    execution.process_entry_fill(
        order_id=order.order_id,
        fill_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert controller.has_position("STUB") is True

    result = controller.evaluate(
        strategy_name="STUB",
        market_data=make_market_data(),
        timestamp=TIMESTAMP,
    )

    assert result.has_position is True
    assert result.decision.action is StrategyAction.EXIT
    assert result.decision.signal is StrategySignal.FLAT
    assert strategy.market_data_calls == 1


def test_notify_fill_calls_strategy_hook():
    execution = make_execution()
    strategy = StubStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    controller.notify_fill(
        strategy_name="STUB",
        market_data=make_market_data(),
    )

    assert strategy.fill_calls == 1


def test_notify_exit_calls_strategy_hook():
    execution = make_execution()
    strategy = StubStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    controller.notify_exit(
        strategy_name="STUB",
    )

    assert strategy.exit_calls == 1


def test_invalid_enter_without_position_is_rejected():
    execution = make_execution()
    strategy = InvalidEntryStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    with pytest.raises(ValueError, match="ENTER decision"):
        controller.evaluate(
            strategy_name="INVALID_ENTRY",
            market_data=make_market_data(),
            timestamp=TIMESTAMP,
        )


def test_invalid_exit_without_position_is_rejected():
    execution = make_execution()
    strategy = InvalidExitFlatStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    with pytest.raises(ValueError, match="cannot emit EXIT"):
        controller.evaluate(
            strategy_name="INVALID_EXIT",
            market_data=make_market_data(),
            timestamp=TIMESTAMP,
        )


def test_invalid_enter_with_active_position_is_rejected():
    execution = make_execution()
    strategy = InvalidActiveEntryStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    intent = make_entry_intent(
        execution,
        strategy_name="INVALID_ACTIVE_ENTRY",
        signal=StrategySignal.LONG,
    )

    order = execution.submit_entry(
        intent,
        quantity=1,
    )

    execution.process_entry_fill(
        order_id=order.order_id,
        fill_price=100.0,
        timestamp=TIMESTAMP,
    )

    with pytest.raises(ValueError, match="cannot emit ENTER"):
        controller.evaluate(
            strategy_name="INVALID_ACTIVE_ENTRY",
            market_data=make_market_data(),
            timestamp=TIMESTAMP,
        )


def test_full_paper_lifecycle_enter_exit():
    execution = make_execution()
    strategy = StubStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    # ENTRY
    entry_result = controller.evaluate(
        strategy_name="STUB",
        market_data=make_market_data(),
        timestamp=TIMESTAMP,
    )

    assert entry_result.decision.action is StrategyAction.ENTER
    assert entry_result.decision.signal is StrategySignal.LONG

    entry_intent = make_entry_intent(
        execution,
        strategy_name="STUB",
        signal=entry_result.decision.signal,
    )

    entry_order = execution.submit_entry(
        entry_intent,
        quantity=1,
    )

    execution.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert execution.get_position("STUB") is not None
    assert controller.has_position("STUB") is True

    controller.notify_fill(
        strategy_name="STUB",
        market_data=make_market_data(),
    )

    assert strategy.fill_calls == 1

    # EXIT
    exit_result = controller.evaluate(
        strategy_name="STUB",
        market_data=make_market_data(),
        timestamp=TIMESTAMP,
    )

    assert exit_result.has_position is True
    assert exit_result.decision.action is StrategyAction.EXIT
    assert exit_result.decision.signal is StrategySignal.FLAT

    exit_intent = make_exit_intent(
        execution,
        strategy_name="STUB",
    )

    exit_order = execution.submit_exit(
        exit_intent,
    )

    execution.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=101.0,
        timestamp=TIMESTAMP,
    )

    assert controller.has_position("STUB") is False

    controller.notify_exit(
        strategy_name="STUB",
    )

    assert strategy.exit_calls == 1


def test_partial_entry_fill_then_update_then_exit():
    execution = make_execution()
    strategy = StubStrategy()

    controller = PositionLifecycleController(
        execution=execution,
        strategies=(strategy,),
    )

    # ENTRY
    entry_intent = make_entry_intent(
        execution,
        strategy_name="STUB",
        signal=StrategySignal.LONG,
    )

    entry_order = execution.submit_entry(
        entry_intent,
        quantity=2,
    )

    # Partial fill 1.
    execution.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=100.0,
        timestamp=TIMESTAMP,
        fill_quantity=1,
    )

    position = execution.get_position("STUB")

    assert position is not None
    assert position.quantity == 1
    assert controller.has_position("STUB") is True

    # Partial fill 2.
    execution.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=102.0,
        timestamp=TIMESTAMP,
        fill_quantity=1,
    )

    position = execution.get_position("STUB")

    assert position is not None
    assert position.quantity == 2

    # ACTIVE POSITION
    result = controller.evaluate(
        strategy_name="STUB",
        market_data=make_market_data(),
        timestamp=TIMESTAMP,
    )

    assert result.has_position is True
    assert result.decision.action is StrategyAction.EXIT

    # EXIT
    exit_intent = make_exit_intent(
        execution,
        strategy_name="STUB",
    )

    exit_order = execution.submit_exit(
        exit_intent,
    )

    execution.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=101.0,
        timestamp=TIMESTAMP,
        fill_quantity=2,
    )

    assert controller.has_position("STUB") is False
