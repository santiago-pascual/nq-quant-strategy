from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)

from .config import S2RConfig
from .fitting import S2FittedModel
from .recovery import (
    RecoveryConfig,
    RecoveryState,
    RecoveryTracker,
)
from .signal import S2SignalRule


class S2RStrategy(BaseStrategy):
    """
    Modular S2R strategy.

    Components:

        S2 frozen signal
            +
        MAE/recovery state machine

    The strategy owns its complete trade lifecycle.

    No fitting, data loading, execution, risk management, or broker logic
    occurs here.
    """

    def __init__(
        self,
        fitted_model: S2FittedModel,
        config: S2RConfig | None = None,
    ) -> None:
        self.config = config or S2RConfig()
        self.fitted_model = fitted_model

        self._entry_rule = S2SignalRule(
            target_state=self.config.target_state,
            quality_threshold=self.config.quality_threshold,
            volatility_low=self.config.volatility_low,
            volatility_high=self.config.volatility_high,
        )

        self._recovery = RecoveryTracker(
            RecoveryConfig(
                mae_threshold_r=self.config.mae_threshold_r,
                recovery_level_r=self.config.recovery_level_r,
                deadline_bars=self.config.recovery_deadline_bars,
            )
        )

        self._in_trade = False
        self._entry_price: float | None = None
        self._entry_bar: int | None = None

    @property
    def name(self) -> str:
        return "S2R"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def in_trade(self) -> bool:
        return self._in_trade

    @property
    def recovery_state(self) -> RecoveryState:
        return self._recovery.state

    @property
    def recovery_decision(self):
        return self._recovery._decision()

    def generate_signal(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategySignal:
        """Generate the frozen S2 short-entry signal."""

        if self._in_trade:
            return StrategySignal.FLAT

        hmm_state = market_data.get("hmm_state")

        if not isinstance(hmm_state, int):
            return StrategySignal.FLAT

        features = {
            feature: market_data.get(feature)
            for feature in self.fitted_model.signal_model.thresholds
        }

        if not self.fitted_model.signal_model.base_signal(features):
            return StrategySignal.FLAT

        quality = self.fitted_model.signal_model.calculate_quality(features)

        volatility_percentile = market_data.get("vol_percentile")

        if volatility_percentile is None:
            realized_vol = market_data.get("realized_vol_30")
            volatility_percentile = self.fitted_model.transform_volatility(realized_vol)

        if self._entry_rule.qualifies(
            hmm_state=hmm_state,
            quality=quality,
            volatility_percentile=volatility_percentile,
        ):
            return StrategySignal.SHORT

        return StrategySignal.FLAT

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        """
        Evaluate both the active recovery state and new-entry conditions.

        Recovery exits have priority over new entries.
        """

        if self._in_trade:
            if self.recovery_state in (
                RecoveryState.RECOVERED,
                RecoveryState.FAILED_TO_RECOVER,
            ):
                return StrategyDecision(
                    signal=StrategySignal.FLAT,
                    action=StrategyAction.EXIT,
                    reason=(
                        f"S2R recovery resolved trade: {self.recovery_state.value}."
                    ),
                )

            return StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.HOLD,
                reason="S2R trade active; recovery is being tracked.",
            )

        signal = self.generate_signal(market_data)

        if signal is StrategySignal.SHORT:
            return StrategyDecision(
                signal=StrategySignal.SHORT,
                action=StrategyAction.ENTER,
                reason="S2 entry conditions satisfied.",
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason="S2 entry conditions not satisfied.",
        )

    def start_trade(
        self,
        *,
        entry_price: float,
        entry_bar: int,
    ) -> None:
        """Start a new S2R trade and reset recovery state."""

        if self._in_trade:
            raise RuntimeError(
                "Cannot start a new trade while S2R is already in a trade."
            )

        self._in_trade = True
        self._entry_price = float(entry_price)
        self._entry_bar = int(entry_bar)

        self._recovery.reset()

    def update_trade(
        self,
        *,
        bar_index: int,
        close_r: float,
        mae_r: float,
    ):
        """Update recovery using already-computed R values."""

        if not self._in_trade:
            raise RuntimeError("Cannot update S2R recovery without an active trade.")

        return self._recovery.update(
            bar_index=bar_index,
            close_r=float(close_r),
            mae_r=float(mae_r),
        )

    def update_trade_from_market(
        self,
        *,
        bar_index: int,
        high: float,
        close: float,
    ):
        """
        Update recovery directly from a completed market bar.

        S2R is short:

            MAE_R   = (high - entry) / stop
            close_R = (entry - close) / stop
        """

        if not self._in_trade:
            raise RuntimeError("Cannot update S2R recovery without an active trade.")

        if self._entry_price is None:
            raise RuntimeError("S2R entry price is missing while a trade is active.")

        stop_points = float(self.config.stop_points)

        if stop_points <= 0:
            raise ValueError("S2R stop_points must be positive.")

        mae_r = (float(high) - self._entry_price) / stop_points
        close_r = (self._entry_price - float(close)) / stop_points

        return self.update_trade(
            bar_index=bar_index,
            close_r=close_r,
            mae_r=mae_r,
        )

    def finish_trade(self) -> None:
        """Clear the completed trade and reset recovery."""

        self._in_trade = False
        self._entry_price = None
        self._entry_bar = None
        self._recovery.reset()
