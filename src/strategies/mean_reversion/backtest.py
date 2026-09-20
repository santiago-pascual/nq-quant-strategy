from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import MeanReversionConfig
from .lifecycle import (
    MeanReversionExit,
    MeanReversionLifecycle,
    MeanReversionTradeState,
)
from .strategy import MeanReversionStrategy
from src.strategies.base import StrategySignal


@dataclass(frozen=True)
class MeanReversionBacktestTrade:
    """Completed Mean Reversion trade."""

    candidate_id: str
    strategy_name: str
    side: str

    # Entry / exit timestamps
    entry_timestamp: Any

    entry_price: float
    exit_price: float

    target_points: float
    stop_points: float
    rr: float
    horizon_bars: int

    exit_reason: str
    bars_elapsed: int
    r_multiple: float


class MeanReversionBacktestAdapter:
    """
    Deterministic historical backtest adapter for Mean Reversion.

    Responsibilities
    ----------------
    - Generate entries using MeanReversionStrategy.
    - Track one active trade.
    - Delegate TP / SL / timeout decisions to MeanReversionLifecycle.
    - Return completed trades.

    It does NOT:
    - optimize parameters
    - load data
    - size positions
    - apply commissions
    - apply slippage
    - interact with a broker
    """

    def __init__(
        self,
        config: MeanReversionConfig,
    ) -> None:
        self.config = config

        self.strategy = MeanReversionStrategy(config)

        self.lifecycle = MeanReversionLifecycle(config)

        self._active_trade: MeanReversionTradeState | None = None

        # Timestamp of the bar on which the active trade entered.
        #
        # This is deliberately kept outside MeanReversionTradeState so that
        # the generic lifecycle state does not need to know about timestamps.
        self._active_entry_timestamp: Any | None = None

    @property
    def active_trade(
        self,
    ) -> MeanReversionTradeState | None:
        """Return the current active trade, if any."""
        return self._active_trade

    def reset(self) -> None:
        """Reset the adapter to a flat state."""
        self._active_trade = None
        self._active_entry_timestamp = None

    def process_bar(
        self,
        market_data: dict,
    ) -> MeanReversionBacktestTrade | None:
        """
        Process one completed market bar.

        Entry
        -----
        If flat, the strategy is evaluated using the supplied
        market context.

        Lifecycle
        ---------
        If a trade is active, OHLC is passed to the lifecycle.

        Important
        ---------
        Entry and exit are mutually exclusive within one call.

        A newly generated entry begins tracking from this bar and
        is not immediately evaluated against its own OHLC.
        """

        # ======================================================
        # ACTIVE TRADE
        # ======================================================

        if self._active_trade is not None:
            exit_result = self.lifecycle.evaluate_bar(
                self._active_trade,
                high=float(market_data["high"]),
                low=float(market_data["low"]),
                close=float(market_data["close"]),
            )

            if exit_result is None:
                self._active_trade = MeanReversionTradeState(
                    entry_price=self._active_trade.entry_price,
                    side=self._active_trade.side,
                    bars_elapsed=(self._active_trade.bars_elapsed + 1),
                )

                return None

            completed = self._build_completed_trade(
                self._active_trade,
                exit_result,
                entry_timestamp=self._active_entry_timestamp,
            )

            self._active_trade = None
            self._active_entry_timestamp = None

            return completed

        # ======================================================
        # FLAT — CHECK ENTRY
        # ======================================================

        signal = self.strategy.generate_signal(market_data)

        if signal is StrategySignal.FLAT:
            return None

        if signal is StrategySignal.LONG:
            side = "LONG"

        elif signal is StrategySignal.SHORT:
            side = "SHORT"

        else:
            raise ValueError(f"Unsupported strategy signal: {signal!r}")

        entry_price = float(market_data["close"])

        self._active_trade = MeanReversionTradeState(
            entry_price=entry_price,
            side=side,
            bars_elapsed=0,
        )

        # Store the exact bar timestamp at entry.
        self._active_entry_timestamp = market_data["timestamp"]

        return None

    def _build_completed_trade(
        self,
        state: MeanReversionTradeState,
        exit_result: MeanReversionExit,
        entry_timestamp: Any | None,
    ) -> MeanReversionBacktestTrade:
        """Convert lifecycle output into a backtest trade."""

        if entry_timestamp is None:
            raise RuntimeError("Completed trade is missing entry timestamp.")

        return MeanReversionBacktestTrade(
            candidate_id=self.config.candidate_id,
            strategy_name=self.config.name,
            side=state.side,
            entry_timestamp=entry_timestamp,
            entry_price=state.entry_price,
            exit_price=exit_result.exit_price,
            target_points=self.config.target_points,
            stop_points=self.config.stop_points,
            rr=self.config.rr,
            horizon_bars=self.config.horizon_bars,
            exit_reason=exit_result.reason.value,
            bars_elapsed=exit_result.bars_elapsed,
            r_multiple=exit_result.r_multiple,
        )

    def run(
        self,
        market_data: list[dict],
    ) -> list[MeanReversionBacktestTrade]:
        """
        Run the adapter over an ordered sequence of bars.

        Any trade still active at the end of the supplied
        dataset remains open and is NOT artificially closed.
        """

        self.reset()

        completed_trades: list[MeanReversionBacktestTrade] = []

        for bar in market_data:
            trade = self.process_bar(bar)

            if trade is not None:
                completed_trades.append(trade)

        return completed_trades
