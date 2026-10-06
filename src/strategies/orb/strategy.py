from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)
from .config import ORBConfig
from .context import ORBContext
from .context import ORBContextBuilder
from .lifecycle import ORBLifecycle, ORBTradeState
from .signal import generate_orb_signal


def is_final_rth_bar(
    market_data: Mapping[str, Any],
    config: ORBConfig,
) -> bool:
    """Check the runner's session-final marker against the configured RTH."""
    if not bool(market_data.get("is_final_rth_bar", False)):
        return False
    timestamp = pd.Timestamp(market_data["timestamp"])
    if timestamp.tzinfo is None:
        raise ValueError("ORB requires timezone-aware timestamps.")
    local_timestamp = timestamp.tz_convert("America/New_York")
    minute = local_timestamp.hour * 60 + local_timestamp.minute
    rth_start = config.rth_start_hour * 60 + config.rth_start_minute
    rth_end = config.rth_end_hour * 60 + config.rth_end_minute
    return rth_start <= minute < rth_end


class ORBStrategy(BaseStrategy):
    """
    Modular Opening Range Breakout strategy.
    """

    SINGLE_CONTRACT_RISK_CAP = 300.0

    def __init__(
        self,
        config: ORBConfig | None = None,
    ) -> None:
        self.config = config or ORBConfig()
        self._context: ORBContext | None = None
        self._context_builder = ORBContextBuilder()
        self._lifecycle = ORBLifecycle(self.config)
        self._trade_state: ORBTradeState | None = None
        self._pending_or_high: float | None = None
        self._pending_or_low: float | None = None
        self._pending_exit_price: float | None = None
        self._in_trade = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def single_contract_risk_cap(self) -> float:
        return self.SINGLE_CONTRACT_RISK_CAP

    @property
    def in_trade(self) -> bool:
        return self._in_trade

    def set_context(self, context: ORBContext) -> None:
        self._context = context

    def reset(self) -> None:
        self._context_builder.reset()
        self._context = None
        self._in_trade = False
        self._trade_state = None
        self._pending_or_high = None
        self._pending_or_low = None
        self._pending_exit_price = None

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        if self._in_trade:
            return StrategySignal.FLAT

        if self._context is None:
            raise RuntimeError(
                "ORB context must be set before generating a signal."
            )

        return generate_orb_signal(
            market_data,
            self._context,
        )

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        self._update_context(market_data)
        signal = self.generate_signal(market_data)

        if signal is StrategySignal.LONG:
            self._remember_entry_range()
            return StrategyDecision(
                signal=signal,
                action=StrategyAction.ENTER,
                reason="ORB upper breakout.",
            )

        if signal is StrategySignal.SHORT:
            self._remember_entry_range()
            return StrategyDecision(
                signal=signal,
                action=StrategyAction.ENTER,
                reason="ORB lower breakout.",
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason="No ORB breakout.",
        )

    def start_trade(self) -> None:
        if self._in_trade:
            raise RuntimeError(
                "Cannot start ORB trade while another trade is active."
            )

        self._in_trade = True

    def finish_trade(self) -> None:
        self._in_trade = False
        self._trade_state = None
        self._pending_or_high = None
        self._pending_or_low = None

    def on_exit(self) -> None:
        self.finish_trade()

    def _update_context(self, market_data: Mapping[str, Any]) -> None:
        timestamp = pd.Timestamp(market_data["timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError("ORB requires timezone-aware timestamps.")
        local_data = dict(market_data)
        local_data["timestamp"] = timestamp.tz_convert("America/New_York").to_pydatetime()
        self._context = self._context_builder.update(
            local_data,
            config=self.config,
        )

    def _remember_entry_range(self) -> None:
        if self._context is None or self._context.or_high is None or self._context.or_low is None:
            raise RuntimeError("ORB entry signal is missing its opening range.")
        self._pending_or_high = float(self._context.or_high)
        self._pending_or_low = float(self._context.or_low)

    def get_entry_fill_price(
        self,
        *,
        signal: StrategySignal,
        market_data: Mapping[str, Any],
    ) -> float | None:
        if signal is StrategySignal.LONG and self._pending_or_high is not None:
            return self._pending_or_high
        if signal is StrategySignal.SHORT and self._pending_or_low is not None:
            return self._pending_or_low
        return None

    def get_entry_reference_price(
        self,
        *,
        signal: StrategySignal,
        market_data: Mapping[str, Any],
    ) -> float:
        fill_price = self.get_entry_fill_price(
            signal=signal,
            market_data=market_data,
        )
        if fill_price is None:
            raise RuntimeError("ORB entry signal has no touched opening-range boundary.")
        return fill_price

    def get_risk_stop_price(
        self,
        *,
        entry_price: float,
        signal: StrategySignal,
        market_data: Mapping[str, Any] | None = None,
    ) -> float | None:
        if signal is StrategySignal.LONG:
            return self._pending_or_low
        if signal is StrategySignal.SHORT:
            return self._pending_or_high
        return None

    def on_fill(
        self,
        *,
        market_data: Mapping[str, Any],
        position: Any,
    ) -> None:
        if position is None or self._pending_or_high is None or self._pending_or_low is None:
            raise RuntimeError("ORB entry fill is missing its opening-range state.")
        if self._trade_state is None:
            self._trade_state = self._lifecycle.create_trade(
                side=position.side.value.upper(),
                entry_price=float(position.entry_price),
                or_high=self._pending_or_high,
                or_low=self._pending_or_low,
            )
            self._in_trade = True
            self._context_builder.mark_trade_taken()

    def on_market_data(
        self,
        market_data: Mapping[str, Any],
        position: Any,
    ) -> StrategyDecision:
        if position is None or self._trade_state is None:
            raise RuntimeError("ORB lifecycle has no active trade state.")
        self._update_context(market_data)
        if self._context is None or not self._context.is_rth:
            return StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.HOLD,
                reason="ORB lifecycle waits for regular-hours bars",
            )
        is_rth_close = is_final_rth_bar(market_data, self.config)
        result = self._lifecycle.evaluate_bar(
            self._trade_state,
            high=float(market_data["high"]),
            low=float(market_data["low"]),
            close=float(market_data["close"]),
            is_rth_close=is_rth_close,
        )
        if result is None:
            self._trade_state = self._lifecycle.advance(self._trade_state)
            return StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.HOLD,
                reason="ORB trade remains active",
            )
        self._pending_exit_price = float(result.exit_price)
        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.EXIT,
            reason=f"ORB {result.reason.value}",
        )

    def get_exit_fill_price(
        self,
        *,
        market_data: Mapping[str, Any],
    ) -> float | None:
        price = self._pending_exit_price
        self._pending_exit_price = None
        return price
