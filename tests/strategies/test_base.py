from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.strategies import BaseStrategy, StrategySignal


class DummyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "0.1.0"

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        direction = market_data.get("direction", "flat")

        if direction == "long":
            return StrategySignal.LONG

        if direction == "short":
            return StrategySignal.SHORT

        return StrategySignal.FLAT


def test_strategy_signal_contains_expected_states() -> None:
    assert StrategySignal.LONG.value == "long"
    assert StrategySignal.SHORT.value == "short"
    assert StrategySignal.FLAT.value == "flat"


def test_base_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseStrategy()


def test_dummy_strategy_can_be_instantiated() -> None:
    strategy = DummyStrategy()

    assert isinstance(strategy, BaseStrategy)


@pytest.mark.parametrize(
    ("market_data", "expected_signal"),
    [
        ({"direction": "long"}, StrategySignal.LONG),
        ({"direction": "short"}, StrategySignal.SHORT),
        ({}, StrategySignal.FLAT),
    ],
)
def test_dummy_strategy_returns_expected_signal(
    market_data: Mapping[str, Any],
    expected_signal: StrategySignal,
) -> None:
    strategy = DummyStrategy()

    assert strategy.generate_signal(market_data) is expected_signal


def test_strategy_metadata_behaves_as_expected() -> None:
    strategy = DummyStrategy()

    assert strategy.name == "dummy"
    assert strategy.version == "0.1.0"


def test_strategy_interface_has_no_external_trading_dependencies() -> None:
    strategy = DummyStrategy()

    signal = strategy.generate_signal({"price": 100.0})

    assert signal is StrategySignal.FLAT
