from __future__ import annotations

from threading import Lock

from .types import (
    BrokerFill,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderStatus,
    BrokerPosition,
)


class BrokerAdapter:
    """
    Provider-neutral broker interface.

    The execution layer depends on this contract, not on Topstep,
    a specific broker, or a specific API.
    """

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def submit_order(
        self,
        request: BrokerOrderRequest,
    ) -> BrokerOrder:
        raise NotImplementedError

    def cancel_order(
        self,
        broker_order_id: str,
    ) -> BrokerOrder:
        raise NotImplementedError

    def get_order(
        self,
        broker_order_id: str,
    ) -> BrokerOrder | None:
        raise NotImplementedError

    def get_positions(self) -> tuple[BrokerPosition, ...]:
        raise NotImplementedError


class InMemoryBrokerAdapter(BrokerAdapter):
    """
    Deterministic broker implementation for development/testing.

    No network, credentials, market-data dependency, or broker-specific
    behavior is included here.
    """

    def __init__(self) -> None:
        self._connected = False
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[BrokerFill] = []
        self._positions: dict[str, BrokerPosition] = {}
        self._order_counter = 0
        self._fill_counter = 0
        self._lock = Lock()

    def connect(self) -> None:
        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def _require_connection(self) -> None:
        if not self.is_connected():
            raise RuntimeError("Broker is not connected")

    def submit_order(
        self,
        request: BrokerOrderRequest,
    ) -> BrokerOrder:
        self._require_connection()

        with self._lock:
            self._order_counter += 1

            broker_order_id = f"SIM-{self._order_counter:08d}"

            order = BrokerOrder(
                broker_order_id=broker_order_id,
                request=request,
                status=BrokerOrderStatus.SUBMITTED,
            )

            self._orders[broker_order_id] = order

            return order

    def cancel_order(
        self,
        broker_order_id: str,
    ) -> BrokerOrder:
        self._require_connection()

        with self._lock:
            order = self._orders.get(broker_order_id)

            if order is None:
                raise KeyError(f"Unknown broker order: {broker_order_id}")

            if order.status in (
                BrokerOrderStatus.FILLED,
                BrokerOrderStatus.CANCELLED,
                BrokerOrderStatus.REJECTED,
            ):
                return order

            cancelled = BrokerOrder(
                broker_order_id=order.broker_order_id,
                request=order.request,
                status=BrokerOrderStatus.CANCELLED,
            )

            self._orders[broker_order_id] = cancelled

            return cancelled

    def get_order(
        self,
        broker_order_id: str,
    ) -> BrokerOrder | None:
        with self._lock:
            return self._orders.get(broker_order_id)

    def get_positions(self) -> tuple[BrokerPosition, ...]:
        with self._lock:
            return tuple(self._positions.values())

    def process_fill(
        self,
        broker_order_id: str,
        quantity: int,
        price: float,
    ) -> BrokerFill:
        self._require_connection()

        with self._lock:
            order = self._orders.get(broker_order_id)

            if order is None:
                raise KeyError(f"Unknown broker order: {broker_order_id}")

            if order.status in (
                BrokerOrderStatus.CANCELLED,
                BrokerOrderStatus.REJECTED,
            ):
                raise ValueError("Cannot fill cancelled or rejected order")

            requested_quantity = order.request.quantity

            existing_quantity = sum(
                fill.quantity
                for fill in self._fills
                if fill.broker_order_id == broker_order_id
            )

            remaining = requested_quantity - existing_quantity

            if quantity <= 0:
                raise ValueError("quantity must be > 0")

            if quantity > remaining:
                raise ValueError("Fill quantity exceeds remaining order quantity")

            self._fill_counter += 1

            fill = BrokerFill(
                broker_order_id=broker_order_id,
                fill_id=f"FILL-{self._fill_counter:08d}",
                quantity=quantity,
                price=price,
                signal=order.request.signal,
            )

            self._fills.append(fill)

            new_filled_quantity = existing_quantity + quantity

            if new_filled_quantity == requested_quantity:
                new_status = BrokerOrderStatus.FILLED
            else:
                new_status = BrokerOrderStatus.PARTIALLY_FILLED

            self._orders[broker_order_id] = BrokerOrder(
                broker_order_id=order.broker_order_id,
                request=order.request,
                status=new_status,
            )

            self._update_position(order, fill)

            return fill

    def _update_position(
        self,
        order: BrokerOrder,
        fill: BrokerFill,
    ) -> None:
        strategy_name = order.request.strategy_name
        symbol = "MNQ.v.0"

        existing = self._positions.get(strategy_name)

        if existing is None:
            self._positions[strategy_name] = BrokerPosition(
                strategy_name=strategy_name,
                symbol=symbol,
                signal=fill.signal,
                quantity=fill.quantity,
                average_price=fill.price,
            )
            return

        if existing.signal == fill.signal:
            total_quantity = existing.quantity + fill.quantity

            total_notional = (
                existing.average_price * existing.quantity + fill.price * fill.quantity
            )

            average_price = total_notional / total_quantity

            self._positions[strategy_name] = BrokerPosition(
                strategy_name=strategy_name,
                symbol=existing.symbol,
                signal=existing.signal,
                quantity=total_quantity,
                average_price=average_price,
            )
            return

        if fill.quantity > existing.quantity:
            raise ValueError("Broker fill quantity exceeds open position quantity")

        remaining = existing.quantity - fill.quantity

        if remaining == 0:
            del self._positions[strategy_name]
        else:
            self._positions[strategy_name] = BrokerPosition(
                strategy_name=strategy_name,
                symbol=existing.symbol,
                signal=existing.signal,
                quantity=remaining,
                average_price=existing.average_price,
            )
