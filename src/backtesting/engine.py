from __future__ import annotations

from dataclasses import dataclass

from src.backtesting.replay import MarketReplay
from src.domain import Fill, MarketData, Position
from src.execution import ExecutionEngine
from src.risk import RiskDecision, RiskManager
from src.strategies.base import BaseStrategy, StrategyDecision


@dataclass(frozen=True)
class BacktestStep:
    """Complete result of processing one market observation."""

    market_data: MarketData
    strategy_decision: StrategyDecision
    risk_decision: RiskDecision
    fills: tuple[Fill, ...]
    position: Position | None


class BacktestEngine:
    """
    Generic deterministic backtesting orchestrator.

    Pipeline:

        MarketReplay
            ↓
        Strategy
            ↓
        RiskManager
            ↓
        ExecutionEngine
            ↓
        Fill
            ↓
        Position

    The engine orchestrates these components but does not implement their
    domain-specific logic.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        risk_manager: RiskManager,
        execution_engine: ExecutionEngine,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine

    def run(
        self,
        replay: MarketReplay,
    ) -> list[BacktestStep]:
        """Run the complete deterministic backtest pipeline."""

        results: list[BacktestStep] = []

        for event in replay:
            market_data = event.data

            self._update_execution_price(market_data)

            position = self.execution_engine.get_position(market_data.symbol)

            strategy_decision = self.strategy.evaluate(
                self._build_market_context(market_data)
            )

            risk_decision = self.risk_manager.evaluate(
                decision=strategy_decision,
                position=position,
            )

            fills: tuple[Fill, ...] = ()

            if risk_decision.approved and risk_decision.order is not None:
                risk_validation = self.risk_manager.validate_order(risk_decision.order)

                if risk_validation.approved:
                    fills = tuple(
                        self.execution_engine.submit_order(risk_decision.order)
                    )

            position = self.execution_engine.get_position(market_data.symbol)

            results.append(
                BacktestStep(
                    market_data=market_data,
                    strategy_decision=strategy_decision,
                    risk_decision=risk_decision,
                    fills=fills,
                    position=position,
                )
            )

        return results

    def _update_execution_price(
        self,
        market_data: MarketData,
    ) -> None:
        """
        Update the execution engine with the current market price.

        BacktestExecutionEngine exposes set_market_price(). Other execution
        implementations may not need this operation.
        """

        setter = getattr(
            self.execution_engine,
            "set_market_price",
            None,
        )

        if setter is not None:
            setter(
                market_data.symbol,
                market_data.close,
            )

    @staticmethod
    def _build_market_context(
        market_data: MarketData,
    ) -> dict:
        """
        Convert MarketData into the generic strategy context.

        Strategy-specific features belong outside the backtest engine.
        """

        return {
            "timestamp": market_data.timestamp,
            "symbol": market_data.symbol,
            "open": market_data.open,
            "high": market_data.high,
            "low": market_data.low,
            "close": market_data.close,
            "volume": market_data.volume,
            "timeframe": market_data.timeframe,
            "bid": market_data.bid,
            "ask": market_data.ask,
        }
