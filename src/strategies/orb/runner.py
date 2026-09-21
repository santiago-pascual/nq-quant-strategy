from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.strategies.base import StrategySignal

from .config import ORBConfig
from .context import ORBContextBuilder
from .lifecycle import ORBExit, ORBLifecycle
from .strategy import ORBStrategy


# =============================================================================
# BACKTEST TRADE
# =============================================================================


@dataclass(frozen=True)
class ORBBacktestTrade:
    strategy_name: str
    strategy_version: str

    side: str

    entry_timestamp: Any
    exit_timestamp: Any

    entry_price: float
    exit_price: float

    stop_price: float
    target_price: float

    opening_range_high: float
    opening_range_low: float
    opening_range_width: float

    rr: float

    exit_reason: str
    bars_elapsed: int

    r_multiple: float


# =============================================================================
# BACKTEST ADAPTER
# =============================================================================


class ORBBacktestAdapter:
    def __init__(
        self,
        config: ORBConfig,
        *,
        rth_close_timestamps: set,
    ) -> None:

        self._config = config

        self._rth_close_timestamps = rth_close_timestamps

        self._context_builder = ORBContextBuilder()

        self._strategy = ORBStrategy(config=config)

        self._lifecycle = ORBLifecycle(config=config)

        self._active_trade = None

        self._entry_timestamp = None
        self._entry_price = None

        self._or_high = None
        self._or_low = None

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:

        self._context_builder.reset()
        self._strategy.reset()

        self._active_trade = None

        self._entry_timestamp = None
        self._entry_price = None

        self._or_high = None
        self._or_low = None

    # =========================================================================
    # PROCESS BAR
    # =========================================================================

    def process_bar(
        self,
        market_data: dict[str, Any],
    ) -> ORBBacktestTrade | None:

        timestamp = market_data["timestamp"]

        # ---------------------------------------------------------------------
        # SESSION BEFORE UPDATE
        # ---------------------------------------------------------------------

        previous_session_date = self._context_builder.session_date

        # ---------------------------------------------------------------------
        # UPDATE CONTEXT
        # ---------------------------------------------------------------------

        context = self._context_builder.update(
            market_data,
            config=self._config,
        )

        # ---------------------------------------------------------------------
        # SESSION CHANGE
        # ---------------------------------------------------------------------

        new_session = (
            previous_session_date is not None
            and context.session_date != previous_session_date
        )

        # ---------------------------------------------------------------------
        # HARD INVARIANT
        #
        # No active ORB trade may survive into another session.
        # ---------------------------------------------------------------------

        if new_session and self._active_trade is not None:
            raise RuntimeError(
                "ORB invariant violated: "
                "active trade survived session boundary. "
                f"Previous session: "
                f"{previous_session_date}; "
                f"new session: "
                f"{context.session_date}; "
                f"entry timestamp: "
                f"{self._entry_timestamp}"
            )

        # ---------------------------------------------------------------------
        # RTH CLOSE
        #
        # IMPORTANT:
        #
        # We do NOT assume 15:59 is always the final RTH bar.
        #
        # The frozen ORB specification exits at the final RTH bar actually
        # present in the market data. This handles holidays / early closes.
        # ---------------------------------------------------------------------

        is_rth_close = timestamp in self._rth_close_timestamps

        # ---------------------------------------------------------------------
        # ACTIVE TRADE
        # ---------------------------------------------------------------------

        if self._active_trade is not None:
            exit_result = self._lifecycle.evaluate_bar(
                self._active_trade,
                high=float(market_data["high"]),
                low=float(market_data["low"]),
                close=float(market_data["close"]),
                is_rth_close=is_rth_close,
            )

            # -------------------------------------------------------------
            # EXIT
            # -------------------------------------------------------------

            if exit_result is not None:
                completed_trade = self._build_completed_trade(
                    timestamp=timestamp,
                    exit_result=exit_result,
                )

                self._active_trade = None

                self._entry_timestamp = None
                self._entry_price = None

                self._or_high = None
                self._or_low = None

                self._strategy.finish_trade()

                return completed_trade

            # -------------------------------------------------------------
            # NO EXIT
            # -------------------------------------------------------------

            self._active_trade = self._lifecycle.advance(self._active_trade)

            return None

        # ---------------------------------------------------------------------
        # NO ACTIVE TRADE
        # ---------------------------------------------------------------------

        self._strategy.set_context(context)

        decision = self._strategy.evaluate(market_data)

        if decision.signal == StrategySignal.FLAT:
            return None

        # ---------------------------------------------------------------------
        # ENTRY
        # ---------------------------------------------------------------------

        if decision.signal == StrategySignal.LONG:
            if context.or_high is None:
                return None

            entry_price = float(context.or_high)

            side = "LONG"

        elif decision.signal == StrategySignal.SHORT:
            if context.or_low is None:
                return None

            entry_price = float(context.or_low)

            side = "SHORT"

        else:
            return None

        # ---------------------------------------------------------------------
        # CREATE TRADE
        # ---------------------------------------------------------------------

        self._active_trade = self._lifecycle.create_trade(
            side=side,
            entry_price=entry_price,
            or_high=float(context.or_high),
            or_low=float(context.or_low),
        )

        self._entry_timestamp = timestamp
        self._entry_price = entry_price

        self._or_high = float(context.or_high)

        self._or_low = float(context.or_low)

        self._strategy.start_trade()

        self._context_builder.mark_trade_taken()

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # The entry bar is NOT evaluated for SL/TP.
        # Management starts from the next bar.
        # ---------------------------------------------------------------------

        return None

    # =========================================================================
    # BUILD COMPLETED TRADE
    # =========================================================================

    def _build_completed_trade(
        self,
        *,
        timestamp,
        exit_result: ORBExit,
    ) -> ORBBacktestTrade:

        if self._active_trade is None:
            raise RuntimeError("Cannot build completed trade without an active trade.")

        if self._entry_timestamp is None:
            raise RuntimeError("Missing entry timestamp.")

        if self._entry_price is None:
            raise RuntimeError("Missing entry price.")

        if self._or_high is None:
            raise RuntimeError("Missing OR high.")

        if self._or_low is None:
            raise RuntimeError("Missing OR low.")

        return ORBBacktestTrade(
            strategy_name=self._strategy.name,
            strategy_version=self._strategy.version,
            side=self._active_trade.side,
            entry_timestamp=self._entry_timestamp,
            exit_timestamp=timestamp,
            entry_price=float(self._entry_price),
            exit_price=float(exit_result.exit_price),
            stop_price=float(self._active_trade.stop_price),
            target_price=float(self._active_trade.target_price),
            opening_range_high=float(self._or_high),
            opening_range_low=float(self._or_low),
            opening_range_width=float(self._or_high - self._or_low),
            rr=float(self._config.rr),
            exit_reason=exit_result.reason.value,
            bars_elapsed=int(exit_result.bars_elapsed),
            r_multiple=float(exit_result.r_multiple),
        )


# =============================================================================
# RUNNER
# =============================================================================


class ORBBacktestRunner:
    REQUIRED_COLUMNS = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    def __init__(
        self,
        config: ORBConfig,
    ) -> None:

        self._config = config

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate_data(data)

        data = data.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="raise",
        )

        data = data.sort_values(
            "timestamp",
            kind="mergesort",
        ).reset_index(drop=True)

        # ---------------------------------------------------------------------
        # IDENTIFY FINAL RTH BAR OF EACH SESSION
        # ---------------------------------------------------------------------
        #
        # This intentionally uses the actual bars present in the input data.
        #
        # Example:
        #
        # normal day:
        #     15:59 ET -> RTH close
        #
        # early-close day:
        #     12:59 ET -> RTH close
        #
        # This matches the frozen ORB semantics:
        #
        #     "exit at final RTH bar close"
        #
        # rather than assuming a fixed 15:59 timestamp.
        # ---------------------------------------------------------------------

        timestamp = data["timestamp"]

        minutes = timestamp.dt.hour * 60 + timestamp.dt.minute

        rth_start_minutes = (
            self._config.rth_start_hour * 60 + self._config.rth_start_minute
        )

        rth_end_minutes = self._config.rth_end_hour * 60 + self._config.rth_end_minute

        rth_mask = (minutes >= rth_start_minutes) & (minutes < rth_end_minutes)

        rth_data = data.loc[
            rth_mask,
            ["timestamp"],
        ]

        if rth_data.empty:
            raise ValueError(
                "ORBBacktestRunner found no RTH bars in the supplied data."
            )

        rth_close_timestamps = set(
            rth_data.groupby(rth_data["timestamp"].dt.date)["timestamp"].max().tolist()
        )

        # ---------------------------------------------------------------------
        # ADAPTER
        # ---------------------------------------------------------------------

        adapter = ORBBacktestAdapter(
            config=self._config,
            rth_close_timestamps=(rth_close_timestamps),
        )

        trades: list[ORBBacktestTrade] = []

        # ---------------------------------------------------------------------
        # BAR LOOP
        # ---------------------------------------------------------------------

        for row in data.itertuples(index=False):
            market_data = {
                "timestamp": row.timestamp,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }

            trade = adapter.process_bar(market_data)

            if trade is not None:
                trades.append(trade)

        # ---------------------------------------------------------------------
        # FINAL INVARIANT
        #
        # A complete dataset should never end while an ORB position remains
        # open. If it does, the input did not contain the final RTH bar.
        # ---------------------------------------------------------------------

        if adapter._active_trade is not None:
            raise RuntimeError(
                "ORB invariant violated: "
                "dataset ended with an active trade. "
                "The supplied data may not contain the "
                "final RTH bar for the last session."
            )

        return self._trades_to_dataframe(trades)

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_data(
        self,
        data: pd.DataFrame,
    ) -> None:

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError("ORBBacktestRunner.run() requires a pandas DataFrame.")

        missing = self.REQUIRED_COLUMNS - set(data.columns)

        if missing:
            raise ValueError(f"Missing required market-data columns: {sorted(missing)}")

        if data.empty:
            raise ValueError("ORBBacktestRunner received an empty DataFrame.")

    # =========================================================================
    # OUTPUT
    # =========================================================================

    @staticmethod
    def _trades_to_dataframe(
        trades: list[ORBBacktestTrade],
    ) -> pd.DataFrame:

        columns = [
            "strategy_name",
            "strategy_version",
            "side",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "stop_price",
            "target_price",
            "opening_range_high",
            "opening_range_low",
            "opening_range_width",
            "rr",
            "exit_reason",
            "bars_elapsed",
            "r_multiple",
        ]

        if not trades:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            [
                {
                    "strategy_name": trade.strategy_name,
                    "strategy_version": trade.strategy_version,
                    "side": trade.side,
                    "entry_timestamp": trade.entry_timestamp,
                    "exit_timestamp": trade.exit_timestamp,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "stop_price": trade.stop_price,
                    "target_price": trade.target_price,
                    "opening_range_high": trade.opening_range_high,
                    "opening_range_low": trade.opening_range_low,
                    "opening_range_width": trade.opening_range_width,
                    "rr": trade.rr,
                    "exit_reason": trade.exit_reason,
                    "bars_elapsed": trade.bars_elapsed,
                    "r_multiple": trade.r_multiple,
                }
                for trade in trades
            ]
        )


# =============================================================================
# FROZEN ENTRY POINT
# =============================================================================


def run_frozen_orb(
    data: pd.DataFrame,
) -> pd.DataFrame:

    config = ORBConfig()

    runner = ORBBacktestRunner(config=config)

    return runner.run(data)
