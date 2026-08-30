from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecoveryState(Enum):
    """Runtime state of the S2R MAE/recovery mechanism."""

    INITIAL = "initial"
    ADVERSE = "adverse"
    RECOVERED = "recovered"
    FAILED_TO_RECOVER = "failed_to_recover"


@dataclass(frozen=True)
class RecoveryConfig:
    """Frozen parameters governing the recovery mechanism."""

    mae_threshold_r: float = 0.70
    recovery_level_r: float = 0.20
    deadline_bars: int = 6


@dataclass(frozen=True)
class RecoveryDecision:
    """Result emitted by the recovery state machine."""

    state: RecoveryState
    mae_bar: int | None
    recovery_bar: int | None
    exit_bar: int | None


class RecoveryModel:
    """
    Deterministic S2R MAE/recovery state machine.

    The model contains no market-data loading, pandas logic, execution
    behavior, account logic, or optimization.

    It receives the observed R-path of the trade and determines whether
    the trade remains in the benchmark path, enters the adverse state,
    recovers within the allowed window, or fails to recover.
    """

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()

    def evaluate(
        self,
        close_r_path: list[float],
        mae_r_path: list[float],
    ) -> RecoveryDecision:
        """
        Evaluate the complete observed trade path.

        Parameters
        ----------
        close_r_path:
            Close-based R values indexed by bar.

        mae_r_path:
            Intrabar MAE R values indexed by bar.

        Returns
        -------
        RecoveryDecision
            Deterministic classification of the trade path.

        Notes
        -----
        The first MAE threshold crossing is authoritative.
        Recovery begins strictly after the MAE bar.
        The recovery deadline is inclusive.
        """

        if len(close_r_path) != len(mae_r_path):
            raise ValueError("close_r_path and mae_r_path must have equal length.")

        if not close_r_path:
            return RecoveryDecision(
                state=RecoveryState.INITIAL,
                mae_bar=None,
                recovery_bar=None,
                exit_bar=None,
            )

        mae_bar = self._find_first_mae_crossing(mae_r_path)

        if mae_bar is None:
            return RecoveryDecision(
                state=RecoveryState.INITIAL,
                mae_bar=None,
                recovery_bar=None,
                exit_bar=None,
            )

        deadline = min(
            mae_bar + self.config.deadline_bars,
            len(close_r_path) - 1,
        )

        recovery_bar = self._find_recovery(
            close_r_path=close_r_path,
            mae_bar=mae_bar,
            deadline=deadline,
        )

        if recovery_bar is not None:
            return RecoveryDecision(
                state=RecoveryState.RECOVERED,
                mae_bar=mae_bar,
                recovery_bar=recovery_bar,
                exit_bar=recovery_bar,
            )

        return RecoveryDecision(
            state=RecoveryState.FAILED_TO_RECOVER,
            mae_bar=mae_bar,
            recovery_bar=None,
            exit_bar=deadline,
        )

    def _find_first_mae_crossing(
        self,
        mae_r_path: list[float],
    ) -> int | None:
        """Return the first bar where MAE reaches the frozen threshold."""

        for bar, mae_r in enumerate(mae_r_path):
            if mae_r >= self.config.mae_threshold_r:
                return bar

        return None

    def _find_recovery(
        self,
        close_r_path: list[float],
        mae_bar: int,
        deadline: int,
    ) -> int | None:
        """
        Find the first recovery bar after MAE.

        The MAE bar itself is excluded.
        The deadline bar itself is included.
        """

        for bar in range(mae_bar + 1, deadline + 1):
            if close_r_path[bar] >= self.config.recovery_level_r:
                return bar

        return None


class RecoveryTracker:
    """
    Incremental S2R recovery state machine.

    Processes one completed bar at a time and never requires future data.
    This is the runtime-oriented counterpart of RecoveryModel.evaluate().
    """

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        self.reset()

    def reset(self) -> None:
        """Reset the tracker to its initial state."""

        self.state = RecoveryState.INITIAL
        self.mae_bar: int | None = None
        self.recovery_bar: int | None = None
        self.exit_bar: int | None = None
        self._last_bar: int | None = None

    def update(
        self,
        *,
        bar_index: int,
        close_r: float,
        mae_r: float,
    ) -> RecoveryDecision:
        """
        Process one completed bar.

        The tracker only uses information available on the current bar.
        """

        if self._last_bar is not None and bar_index <= self._last_bar:
            raise ValueError("bar_index must increase strictly between updates.")

        self._last_bar = bar_index

        if self.state is RecoveryState.INITIAL:
            if mae_r >= self.config.mae_threshold_r:
                self.state = RecoveryState.ADVERSE
                self.mae_bar = bar_index

            return self._decision()

        if self.state is RecoveryState.ADVERSE:
            assert self.mae_bar is not None

            # Recovery cannot occur on the MAE bar itself.
            if bar_index > self.mae_bar and close_r >= self.config.recovery_level_r:
                self.state = RecoveryState.RECOVERED
                self.recovery_bar = bar_index
                self.exit_bar = bar_index
                return self._decision()

            deadline = self.mae_bar + self.config.deadline_bars

            if bar_index >= deadline:
                self.state = RecoveryState.FAILED_TO_RECOVER
                self.exit_bar = deadline
                return self._decision()

            return self._decision()

        # Terminal states remain terminal.
        return self._decision()

    def _decision(self) -> RecoveryDecision:
        """Build an immutable snapshot of the current tracker state."""

        return RecoveryDecision(
            state=self.state,
            mae_bar=self.mae_bar,
            recovery_bar=self.recovery_bar,
            exit_bar=self.exit_bar,
        )
