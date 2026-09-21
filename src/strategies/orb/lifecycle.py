from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ORBConfig, ORBExitReason


@dataclass(frozen=True)
class ORBTradeState:
    entry_price: float
    side: str
    stop_price: float
    target_price: float
    bars_elapsed: int = 0


@dataclass(frozen=True)
class ORBExit:
    reason: ORBExitReason
    exit_price: float
    bars_elapsed: int
    r_multiple: float


class ORBLifecycle:
    """
    Deterministic lifecycle for an active ORB trade.

    Exit priority:

        1. STOP
        2. TARGET
        3. RTH CLOSE

    STOP is intentionally checked before TARGET when both are touched
    by the same candle.
    """

    def __init__(self, config: ORBConfig) -> None:
        self.config = config

    def create_trade(
        self,
        *,
        side: str,
        entry_price: float,
        or_high: float,
        or_low: float,
    ) -> ORBTradeState:
        entry_price = float(entry_price)
        or_high = float(or_high)
        or_low = float(or_low)

        opening_range_width = or_high - or_low

        if opening_range_width <= 0:
            raise ValueError("ORB opening range width must be positive.")

        if side == "LONG":
            stop_price = or_low
            risk = entry_price - stop_price

            if risk <= 0:
                raise ValueError("Invalid LONG ORB risk.")

            target_price = entry_price + (risk * self.config.rr)

        elif side == "SHORT":
            stop_price = or_high
            risk = stop_price - entry_price

            if risk <= 0:
                raise ValueError("Invalid SHORT ORB risk.")

            target_price = entry_price - (risk * self.config.rr)

        else:
            raise ValueError(f"Unsupported ORB side: {side!r}")

        return ORBTradeState(
            entry_price=entry_price,
            side=side,
            stop_price=float(stop_price),
            target_price=float(target_price),
            bars_elapsed=0,
        )

    def evaluate_bar(
        self,
        state: ORBTradeState,
        *,
        high: float,
        low: float,
        close: float,
        is_rth_close: bool,
    ) -> ORBExit | None:
        high = float(high)
        low = float(low)
        close = float(close)

        bars_elapsed = state.bars_elapsed + 1

        if state.side == "LONG":
            stop_hit = low <= state.stop_price
            target_hit = high >= state.target_price

            # Conservative intrabar ambiguity:
            # stop has priority over target.
            if stop_hit:
                return ORBExit(
                    reason=ORBExitReason.STOP,
                    exit_price=state.stop_price,
                    bars_elapsed=bars_elapsed,
                    r_multiple=-1.0,
                )

            if target_hit:
                return ORBExit(
                    reason=ORBExitReason.TARGET,
                    exit_price=state.target_price,
                    bars_elapsed=bars_elapsed,
                    r_multiple=self.config.rr,
                )

        elif state.side == "SHORT":
            stop_hit = high >= state.stop_price
            target_hit = low <= state.target_price

            if stop_hit:
                return ORBExit(
                    reason=ORBExitReason.STOP,
                    exit_price=state.stop_price,
                    bars_elapsed=bars_elapsed,
                    r_multiple=-1.0,
                )

            if target_hit:
                return ORBExit(
                    reason=ORBExitReason.TARGET,
                    exit_price=state.target_price,
                    bars_elapsed=bars_elapsed,
                    r_multiple=self.config.rr,
                )

        else:
            raise ValueError(f"Unsupported ORB side: {state.side!r}")

        # RTH close is the final exit condition.
        if is_rth_close:
            if state.side == "LONG":
                risk = state.entry_price - state.stop_price
                r_multiple = (close - state.entry_price) / risk
            else:
                risk = state.stop_price - state.entry_price
                r_multiple = (state.entry_price - close) / risk

            return ORBExit(
                reason=ORBExitReason.RTH_CLOSE,
                exit_price=close,
                bars_elapsed=bars_elapsed,
                r_multiple=float(r_multiple),
            )

        return None

    def advance(
        self,
        state: ORBTradeState,
    ) -> ORBTradeState:
        return ORBTradeState(
            entry_price=state.entry_price,
            side=state.side,
            stop_price=state.stop_price,
            target_price=state.target_price,
            bars_elapsed=state.bars_elapsed + 1,
        )
