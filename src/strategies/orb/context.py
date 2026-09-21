from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ORBContext:
    session_date: date | None

    or_high: float | None
    or_low: float | None
    or_bars: int
    or_ready: bool

    is_rth: bool
    entry_window_open: bool

    trade_taken: bool

    # True once a single entry-window bar has touched
    # both OR boundaries. The entire session is then
    # invalidated for ORB trading.
    session_invalidated: bool


class ORBContextBuilder:
    def __init__(self) -> None:
        self.reset()

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:

        self._session_date: date | None = None

        self._or_high: float | None = None
        self._or_low: float | None = None

        self._or_bars = 0

        self._trade_taken = False

        self._session_invalidated = False

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def session_date(self) -> date | None:
        return self._session_date

    @property
    def or_high(self) -> float | None:
        return self._or_high

    @property
    def or_low(self) -> float | None:
        return self._or_low

    @property
    def or_bars(self) -> int:
        return self._or_bars

    @property
    def trade_taken(self) -> bool:
        return self._trade_taken

    @property
    def session_invalidated(self) -> bool:
        return self._session_invalidated

    # =========================================================================
    # TRADE STATE
    # =========================================================================

    def mark_trade_taken(self) -> None:
        self._trade_taken = True

    # =========================================================================
    # SESSION INVALIDATION
    # =========================================================================

    def invalidate_session(self) -> None:
        self._session_invalidated = True

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(
        self,
        market_data: dict[str, Any],
        *,
        config,
    ) -> ORBContext:

        timestamp = market_data["timestamp"]

        if timestamp.tzinfo is None:
            raise ValueError("ORB requires timezone-aware timestamps.")

        current_date = timestamp.date()

        # ---------------------------------------------------------------------
        # NEW SESSION
        # ---------------------------------------------------------------------

        if self._session_date != current_date:
            self.reset()

            self._session_date = current_date

        # ---------------------------------------------------------------------
        # CLOCK
        # ---------------------------------------------------------------------

        current_minutes = timestamp.hour * 60 + timestamp.minute

        rth_start = config.rth_start_hour * 60 + config.rth_start_minute

        or_end = config.opening_range_end_hour * 60 + config.opening_range_end_minute

        entry_cutoff = config.entry_cutoff_hour * 60 + config.entry_cutoff_minute

        rth_end = config.rth_end_hour * 60 + config.rth_end_minute

        # ---------------------------------------------------------------------
        # RTH
        # ---------------------------------------------------------------------

        is_rth = rth_start <= current_minutes < rth_end

        # ---------------------------------------------------------------------
        # OPENING RANGE
        # ---------------------------------------------------------------------

        if rth_start <= current_minutes < or_end:
            high = float(market_data["high"])

            low = float(market_data["low"])

            self._or_bars += 1

            if self._or_high is None:
                self._or_high = high
            else:
                self._or_high = max(
                    self._or_high,
                    high,
                )

            if self._or_low is None:
                self._or_low = low
            else:
                self._or_low = min(
                    self._or_low,
                    low,
                )

        # ---------------------------------------------------------------------
        # OR READY
        # ---------------------------------------------------------------------

        or_ready = (
            self._or_bars == 30
            and self._or_high is not None
            and self._or_low is not None
            and current_minutes >= or_end
        )

        # ---------------------------------------------------------------------
        # ENTRY WINDOW
        # ---------------------------------------------------------------------

        entry_window_open = (
            or_ready
            and is_rth
            and or_end <= current_minutes < entry_cutoff
            and not self._trade_taken
            and not self._session_invalidated
        )

        # ---------------------------------------------------------------------
        # SAME-BAR DOUBLE BREAKOUT
        # ---------------------------------------------------------------------
        #
        # Frozen ORB rule:
        #
        # If a single entry-window candle touches both OR boundaries,
        # the ENTIRE SESSION is skipped.
        #
        # This must happen before signal generation on the same bar.
        # ---------------------------------------------------------------------

        if (
            or_ready
            and is_rth
            and or_end <= current_minutes < entry_cutoff
            and self._or_high is not None
            and self._or_low is not None
            and not self._trade_taken
            and not self._session_invalidated
        ):
            high = float(market_data["high"])

            low = float(market_data["low"])

            touches_high = high >= self._or_high

            touches_low = low <= self._or_low

            if touches_high and touches_low:
                self._session_invalidated = True

                entry_window_open = False

        # ---------------------------------------------------------------------
        # RETURN SNAPSHOT
        # ---------------------------------------------------------------------

        return ORBContext(
            session_date=self._session_date,
            or_high=self._or_high,
            or_low=self._or_low,
            or_bars=self._or_bars,
            or_ready=or_ready,
            is_rth=is_rth,
            entry_window_open=entry_window_open,
            trade_taken=self._trade_taken,
            session_invalidated=(self._session_invalidated),
        )
