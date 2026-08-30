from __future__ import annotations

from dataclasses import dataclass

from src.backtesting.replay import MarketReplay
from src.domain import Fill, MarketData, Position
from src.execution import ExecutionEngine
from src.risk import RiskDecision, RiskManager
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
)


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

    Strategy-specific behavior remains inside the strategy.

    The engine only orchestrates:

        replay
          ↓
        strategy
          ↓
        risk
          ↓
        execution
          ↓
        position
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

        for bar_index, event in enumerate(replay):
            market_data = event.data

            self._update_execution_price(market_data)

            context = self._build_market_context(
                market_data,
                bar_index=bar_index,
            )

            position = self.execution_engine.get_position(market_data.symbol)

            # --------------------------------------------------------
            # ACTIVE TRADE
            # --------------------------------------------------------

            if position is not None:
                strategy_exit = self.strategy.on_market_data(
                    context,
                    position,
                )

                if (
                    strategy_exit is not None
                    and strategy_exit.action is StrategyAction.EXIT
                ):
                    risk_decision = self.risk_manager.evaluate(
                        decision=strategy_exit,
                        position=position,
                    )

                    fills: tuple[Fill, ...] = ()

                    if risk_decision.approved:
                        if risk_decision.order is not None:
                            risk_validation = self.risk_manager.validate_order(
                                risk_decision.order
                            )

                            if risk_validation.approved:
                                fills = tuple(
                                    self.execution_engine.submit_order(
                                        risk_decision.order
                                    )
                                )
                        else:
                            fills = tuple(
                                self.execution_engine.close_position(market_data.symbol)
                            )

                    position = self.execution_engine.get_position(market_data.symbol)

                    if not position:
                        on_exit = getattr(
                            self.strategy,
                            "on_exit",
                            None,
                        )

                        if on_exit is not None and fills:
                            on_exit()

                    results.append(
                        BacktestStep(
                            market_data=market_data,
                            strategy_decision=strategy_exit,
                            risk_decision=risk_decision,
                            fills=fills,
                            position=position,
                        )
                    )

                    continue

            # --------------------------------------------------------
            # NORMAL STRATEGY EVALUATION
            # --------------------------------------------------------

            strategy_decision = self.strategy.evaluate(context)

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

            # --------------------------------------------------------
            # ENTRY FILL → STRATEGY LIFECYCLE
            # --------------------------------------------------------

            if fills and position is not None:
                self.strategy.on_fill(
                    market_data=context,
                    position=position,
                )

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
        """Update execution engine with the current replay price."""

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
        *,
        bar_index: int,
    ) -> dict:
        """
        Convert MarketData into the generic strategy context.

        bar_index belongs to the replay lifecycle, not MarketData itself.
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
            "bar_index": bar_index,
        }
