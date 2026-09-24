from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ConflictDecision(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PortfolioPosition:
    """
    Representation of an already-open strategy position.

    This is deliberately independent from broker/execution objects.
    Portfolio conflict logic only needs strategy identity and direction.
    """

    strategy_name: str
    side: str
    quantity: int = 1


@dataclass(frozen=True)
class EntryRequest:
    """
    Portfolio-level request to open a new strategy position.
    """

    strategy_name: str
    side: str
    quantity: int = 1


@dataclass(frozen=True)
class ConflictResult:
    decision: ConflictDecision
    strategy_name: str
    reason: str

    @property
    def approved(self) -> bool:
        return self.decision is ConflictDecision.APPROVED


class PortfolioConflictEngine:
    """
    Deterministic portfolio-level conflict resolver.

    Responsibilities
    -----------------
    - Prevent duplicate positions for the same strategy.
    - Enforce the maximum number of concurrent strategy positions.
    - Resolve simultaneous entries deterministically.
    - Reject malformed/invalid requests.
    - Keep portfolio conflict logic separate from risk sizing
      and broker/execution mechanics.

    Non-responsibilities
    --------------------
    - Position sizing.
    - Dollar risk calculations.
    - Stop/target calculations.
    - Broker communication.
    - Order submission.
    - PnL accounting.
    """

    def __init__(
        self,
        *,
        max_concurrent_positions: int,
    ) -> None:
        if max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be greater than zero")

        self.max_concurrent_positions = max_concurrent_positions

        self._positions: dict[str, PortfolioPosition] = {}

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    @property
    def open_strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self._positions))

    def has_open_position(self, strategy_name: str) -> bool:
        return strategy_name in self._positions

    def get_position(
        self,
        strategy_name: str,
    ) -> PortfolioPosition | None:
        return self._positions.get(strategy_name)

    def reset(self) -> None:
        self._positions.clear()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entry_request(request: EntryRequest) -> None:
        if not request.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if request.side not in {"long", "short"}:
            raise ValueError("side must be either 'long' or 'short'")

        if request.quantity <= 0:
            raise ValueError("quantity must be greater than zero")

    # ------------------------------------------------------------------
    # Single-entry evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        request: EntryRequest,
    ) -> ConflictResult:
        """
        Evaluate a single entry without mutating portfolio state.
        """

        self._validate_entry_request(request)

        if self.has_open_position(request.strategy_name):
            return ConflictResult(
                decision=ConflictDecision.REJECTED,
                strategy_name=request.strategy_name,
                reason=("strategy already has an open portfolio position"),
            )

        if self.open_position_count >= self.max_concurrent_positions:
            return ConflictResult(
                decision=ConflictDecision.REJECTED,
                strategy_name=request.strategy_name,
                reason=("maximum concurrent portfolio positions reached"),
            )

        return ConflictResult(
            decision=ConflictDecision.APPROVED,
            strategy_name=request.strategy_name,
            reason="portfolio conflict checks passed",
        )

    # ------------------------------------------------------------------
    # Commit / release
    # ------------------------------------------------------------------

    def register_entry(
        self,
        request: EntryRequest,
    ) -> PortfolioPosition:
        """
        Commit an approved entry to portfolio state.

        This must happen only after the corresponding execution layer
        has confirmed the entry fill.
        """

        result = self.evaluate(request)

        if not result.approved:
            raise RuntimeError(result.reason)

        position = PortfolioPosition(
            strategy_name=request.strategy_name,
            side=request.side,
            quantity=request.quantity,
        )

        self._positions[request.strategy_name] = position

        return position

    def register_exit(
        self,
        strategy_name: str,
    ) -> PortfolioPosition:
        """
        Remove a strategy from the active portfolio.

        The closed position is returned so callers can audit the
        lifecycle transition.
        """

        if not strategy_name:
            raise ValueError("strategy_name must not be empty")

        position = self._positions.pop(strategy_name, None)

        if position is None:
            raise RuntimeError("strategy has no open portfolio position")

        return position

    # ------------------------------------------------------------------
    # Batch / simultaneous entry resolution
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        requests: Iterable[EntryRequest],
    ) -> list[ConflictResult]:
        """
        Evaluate multiple candidate entries in deterministic order.

        The evaluation itself does not mutate portfolio state.

        Deterministic ordering:
            1. strategy_name
            2. side
            3. quantity

        This prevents result changes caused solely by the order in which
        upstream strategies happened to emit their signals.
        """

        normalized = list(requests)

        for request in normalized:
            self._validate_entry_request(request)

        ordered = sorted(
            normalized,
            key=lambda request: (
                request.strategy_name,
                request.side,
                request.quantity,
            ),
        )

        results: list[ConflictResult] = []

        seen_strategies: set[str] = set()
        available_slots = self.max_concurrent_positions - self.open_position_count

        for request in ordered:
            if request.strategy_name in self._positions:
                results.append(
                    ConflictResult(
                        decision=ConflictDecision.REJECTED,
                        strategy_name=request.strategy_name,
                        reason=("strategy already has an open portfolio position"),
                    )
                )
                continue

            if request.strategy_name in seen_strategies:
                results.append(
                    ConflictResult(
                        decision=ConflictDecision.REJECTED,
                        strategy_name=request.strategy_name,
                        reason=("multiple simultaneous entries for the same strategy"),
                    )
                )
                continue

            if available_slots <= 0:
                results.append(
                    ConflictResult(
                        decision=ConflictDecision.REJECTED,
                        strategy_name=request.strategy_name,
                        reason=("maximum concurrent portfolio positions reached"),
                    )
                )
                continue

            results.append(
                ConflictResult(
                    decision=ConflictDecision.APPROVED,
                    strategy_name=request.strategy_name,
                    reason="portfolio conflict checks passed",
                )
            )

            seen_strategies.add(request.strategy_name)
            available_slots -= 1

        return results

    def register_batch(
        self,
        requests: Iterable[EntryRequest],
    ) -> list[PortfolioPosition]:
        """
        Atomically evaluate and register a batch of entries.

        If any request is rejected, no position is registered.
        """

        normalized = list(requests)
        results = self.evaluate_batch(normalized)

        rejected = [result for result in results if not result.approved]

        if rejected:
            reasons = "; ".join(
                f"{result.strategy_name}: {result.reason}" for result in rejected
            )

            raise RuntimeError(f"portfolio batch rejected: {reasons}")

        ordered = sorted(
            normalized,
            key=lambda request: (
                request.strategy_name,
                request.side,
                request.quantity,
            ),
        )

        positions: list[PortfolioPosition] = []

        for request in ordered:
            position = PortfolioPosition(
                strategy_name=request.strategy_name,
                side=request.side,
                quantity=request.quantity,
            )

            self._positions[request.strategy_name] = position
            positions.append(position)

        return positions
