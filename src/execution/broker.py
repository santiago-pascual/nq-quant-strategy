from abc import ABC, abstractmethod

from .types import Fill, Order


class BrokerAdapter(ABC):
    """
    Interface between the execution engine and an external
    execution venue.

    The execution engine must not depend on a specific broker API.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> None:
        """Submit an order to the execution venue."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Request cancellation of an existing order."""
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str):
        """Return the broker-side status of an order."""
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Return broker-side open orders."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list:
        """Return broker-side positions."""
        raise NotImplementedError

    @abstractmethod
    def get_fills(self) -> list[Fill]:
        """Return known fills."""
        raise NotImplementedError
