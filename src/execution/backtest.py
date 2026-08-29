from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.domain import Fill, Order, OrderSide, Position, PositionSide
from src.execution.base import ExecutionEngine
from src.infrastructure import Clock, FixedClock


class BacktestExecutionEngine(ExecutionEngine):
    """
    Deterministic execution engine for historical backtests.

    Market orders are filled at the current supplied market price.

    Execution is driven by an explicit Clock so that backtests and replay
    runs remain deterministic and reproducible.
    """

    def __init__(
        self,
        market_prices: dict[str, float],
        clock: Clock,
        commission_per_unit: float = 0.0,
        slippage_per_unit: float = 0.0,
    ) -> None:
        self._market_prices = dict(market_prices)
        self._clock = clock
        self._commission_per_unit = commission_per_unit
        self._slippage_per_unit = slippage_per_unit
        self._positions: dict[str, Position] = {}

    @property
    def name(self) -> str:
        return "BacktestExecutionEngine"

    @property
    def mode(self) -> str:
        return "backtest"

    def submit_order(self, order: Order) -> Sequence[Fill]:
        """Execute an order deterministically at the current market price."""

        if order.symbol not in self._market_prices:
            raise ValueError(f"No market price available for symbol: {order.symbol}")

        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive.")

        if order.order_type.value != "market":
            raise NotImplementedError(
                "BacktestExecutionEngine currently supports MARKET orders only."
            )

        market_price = self._market_prices[order.symbol]

        execution_price = self._apply_slippage(
            side=order.side,
            price=market_price,
        )

        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=self._clock.now(),
            order_id=order.client_order_id,
            commission=order.quantity * self._commission_per_unit,
            slippage=abs(execution_price - market_price) * order.quantity,
        )

        self._apply_fill(fill)

        return [fill]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an outstanding order."""

        return False

    def get_position(self, symbol: str) -> Position | None:
        """Return the current position for a symbol."""

        return self._positions.get(symbol)

    def close_position(self, symbol: str) -> Sequence[Fill]:
        """Close the current position using the current market price."""

        position = self._positions.get(symbol)

        if position is None:
            return []

        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY

        order = Order(
            symbol=symbol,
            side=side,
            quantity=position.quantity,
        )

        return self.submit_order(order)

    def set_market_price(
        self,
        symbol: str,
        price: float,
    ) -> None:
        """Update the current market price."""

        if price <= 0:
            raise ValueError("Market price must be positive.")

        self._market_prices[symbol] = price

    def _apply_slippage(
        self,
        side: OrderSide,
        price: float,
    ) -> float:
        """Apply deterministic adverse slippage."""

        if side is OrderSide.BUY:
            return price + self._slippage_per_unit

        return price - self._slippage_per_unit

    def _apply_fill(self, fill: Fill) -> None:
        """Update internal position state from an executed fill."""

        current = self._positions.get(fill.symbol)

        if current is None:
            side = (
                PositionSide.LONG if fill.side is OrderSide.BUY else PositionSide.SHORT
            )

            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                side=side,
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_timestamp=fill.timestamp,
            )
            return

        if current.side is PositionSide.LONG and fill.side is OrderSide.BUY:
            new_quantity = current.quantity + fill.quantity

            new_entry_price = (
                current.entry_price * current.quantity + fill.price * fill.quantity
            ) / new_quantity

            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                side=current.side,
                quantity=new_quantity,
                entry_price=new_entry_price,
                entry_timestamp=current.entry_timestamp,
            )
            return

        if current.side is PositionSide.SHORT and fill.side is OrderSide.SELL:
            new_quantity = current.quantity + fill.quantity

            new_entry_price = (
                current.entry_price * current.quantity + fill.price * fill.quantity
            ) / new_quantity

            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                side=current.side,
                quantity=new_quantity,
                entry_price=new_entry_price,
                entry_timestamp=current.entry_timestamp,
            )
            return

        if fill.quantity < current.quantity:
            remaining_quantity = current.quantity - fill.quantity

            self._positions[fill.symbol] = Position(
                symbol=current.symbol,
                side=current.side,
                quantity=remaining_quantity,
                entry_price=current.entry_price,
                entry_timestamp=current.entry_timestamp,
            )
            return

        if fill.quantity == current.quantity:
            del self._positions[fill.symbol]
            return

        raise ValueError("Fill quantity cannot reverse an existing position.")
