from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import MeanReversionConfig


class MeanReversionExitReason(Enum):
    """Reason why a Mean Reversion trade finished."""

    TARGET = "target"
    STOP = "stop"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class MeanReversionTradeState:
    """
    Immutable state of one active Mean Reversion trade.

    Price distances are expressed in MNQ points.
    """

    entry_price: float
    side: str
    bars_elapsed: int = 0


@dataclass(frozen=True)
class MeanReversionExit:
    """Deterministic result of evaluating one active trade."""

    reason: MeanReversionExitReason
    exit_price: float
    r_multiple: float
    bars_elapsed: int


class MeanReversionLifecycle:
    """
    Deterministic TP / SL / horizon evaluator.

    This component does not:
        - generate entries
        - load market data
        - optimize parameters
        - size positions
        - execute broker orders

    It only manages the payoff geometry of an active trade.
    """

    def __init__(
        self,
        config: MeanReversionConfig,
    ) -> None:
        self.config = config

    def evaluate_bar(
        self,
        state: MeanReversionTradeState,
        *,
        high: float,
        low: float,
        close: float,
    ) -> MeanReversionExit | None:
        """
        Evaluate one completed market bar.

        Barrier policy:
            - TP and SL are checked using OHLC.
            - If both barriers are touched on the same bar,
              STOP is authoritative.
            - If neither barrier is hit and the horizon expires,
              exit at the bar close.
        """

        if state.side not in {"LONG", "SHORT"}:
            raise ValueError(f"Unsupported Mean Reversion side: {state.side!r}")

        if high < low:
            raise ValueError("high cannot be below low.")

        if state.entry_price <= 0:
            raise ValueError("entry_price must be positive.")

        next_bar = state.bars_elapsed + 1

        if state.side == "LONG":
            target_price = state.entry_price + self.config.target_points

            stop_price = state.entry_price - self.config.stop_points

            target_hit = high >= target_price
            stop_hit = low <= stop_price

            # Same-bar conflict: STOP FIRST.
            if stop_hit:
                return MeanReversionExit(
                    reason=MeanReversionExitReason.STOP,
                    exit_price=stop_price,
                    r_multiple=-1.0,
                    bars_elapsed=next_bar,
                )

            if target_hit:
                return MeanReversionExit(
                    reason=MeanReversionExitReason.TARGET,
                    exit_price=target_price,
                    r_multiple=self.config.rr,
                    bars_elapsed=next_bar,
                )

            if next_bar >= self.config.horizon_bars:
                movement = close - state.entry_price

                r_multiple = movement / self.config.stop_points

                return MeanReversionExit(
                    reason=MeanReversionExitReason.TIMEOUT,
                    exit_price=close,
                    r_multiple=r_multiple,
                    bars_elapsed=next_bar,
                )

            return None

        # SHORT
        target_price = state.entry_price - self.config.target_points

        stop_price = state.entry_price + self.config.stop_points

        target_hit = low <= target_price
        stop_hit = high >= stop_price

        # Same-bar conflict: STOP FIRST.
        if stop_hit:
            return MeanReversionExit(
                reason=MeanReversionExitReason.STOP,
                exit_price=stop_price,
                r_multiple=-1.0,
                bars_elapsed=next_bar,
            )

        if target_hit:
            return MeanReversionExit(
                reason=MeanReversionExitReason.TARGET,
                exit_price=target_price,
                r_multiple=self.config.rr,
                bars_elapsed=next_bar,
            )

        if next_bar >= self.config.horizon_bars:
            movement = state.entry_price - close

            r_multiple = movement / self.config.stop_points

            return MeanReversionExit(
                reason=MeanReversionExitReason.TIMEOUT,
                exit_price=close,
                r_multiple=r_multiple,
                bars_elapsed=next_bar,
            )

        return None
