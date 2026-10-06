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
        fitted_model: S2FittedModel | None = None,
        config: S2RConfig | None = None,
    ) -> None:
        self.config = config or S2RConfig()
        self.fitted_model = fitted_model
        self.model_window: int | None = None
        self._window_models_for_bar: dict[int, S2FittedModel] = {}
        self._window_evaluations: list[dict[str, Any]] = []

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
        self._last_bar_index = 0
        self._pending_exit_price: float | None = None

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

        self._window_evaluations = []
        if self._in_trade:
            return StrategySignal.FLAT
        models = self._window_models_for_bar or (
            {self.model_window: self.fitted_model}
            if self.fitted_model is not None
            else {}
        )
        if "s2r_entry_eligible" in market_data and not bool(
            market_data["s2r_entry_eligible"]
        ):
            for window, _model in sorted(models.items()):
                if "s2r_hmm_states" in market_data:
                    hmm_state = dict(market_data["s2r_hmm_states"]).get(window)
                else:
                    hmm_state = market_data.get("hmm_state")
                self._window_evaluations.append(
                    {
                        "window": window,
                        "hmm_state": hmm_state,
                        "qualifies": False,
                        "reason": "research_horizon_cutoff",
                        "quality": None,
                        "volatility_percentile": None,
                    }
                )
            return StrategySignal.FLAT
        if not models:
            return StrategySignal.FLAT

        # The Research window model changes the entry gate only. All qualifying
        # evaluations at this one bar share the fixed S2R short side, session,
        # and lifecycle, so they produce one economic entry intent. Window
        # identity remains available in the per-window evaluation diagnostics.
        qualifies_any = False
        for window, model in sorted(models.items()):
            evaluation = self.evaluate_entry_window(
                market_data,
                model=model,
                window=window,
            )
            self._window_evaluations.append(evaluation)
            qualifies_any = qualifies_any or bool(evaluation["qualifies"])
        return StrategySignal.SHORT if qualifies_any else StrategySignal.FLAT

    def evaluate_entry_window(
        self,
        market_data: Mapping[str, Any],
        *,
        model: S2FittedModel,
        window: int | None,
    ) -> dict[str, Any]:
        """Evaluate one frozen Research window without creating an order."""
        # Research Replay supplies a distinct state prediction for each
        # window-local S2R HMM. If that mapping is present, a missing window
        # state must not fall back to the shared MRL/MRS HMM provider.
        if "s2r_hmm_states" in market_data:
            hmm_state = dict(market_data["s2r_hmm_states"]).get(window)
        else:
            # Keep direct strategy use and existing callers provider-agnostic.
            hmm_state = market_data.get("hmm_state")
        if not isinstance(hmm_state, int):
            return {"window": window, "hmm_state": hmm_state, "qualifies": False,
                    "reason": "missing_target_state", "quality": None,
                    "volatility_percentile": None}
        features = {feature: market_data.get(feature)
                    for feature in model.signal_model.thresholds}
        if not model.signal_model.base_signal(features):
            return {"window": window, "hmm_state": hmm_state, "qualifies": False,
                    "reason": "base_signal_failed", "quality": None,
                    "volatility_percentile": None}
        quality = model.signal_model.calculate_quality(features)
        realized_vol = market_data.get("realized_vol_30")
        volatility_percentile = model.transform_volatility(realized_vol)
        qualifies = self._entry_rule.qualifies(
            hmm_state=hmm_state,
            quality=quality,
            volatility_percentile=volatility_percentile,
        )
        if qualifies:
            reason = "qualified"
        elif hmm_state != self.config.target_state:
            reason = "target_state_mismatch"
        elif quality is None or quality < self.config.quality_threshold:
            reason = "quality_threshold_not_met"
        else:
            reason = "volatility_percentile_outside_window"
        return {
            "window": window,
            "hmm_state": hmm_state,
            "qualifies": bool(qualifies),
            "reason": reason,
            "quality": None if quality is None else float(quality),
            "volatility_percentile": (
                None
                if volatility_percentile is None
                else float(volatility_percentile)
            ),
        }

    def set_fitted_models_for_bar(
        self,
        models: Mapping[int, S2FittedModel],
    ) -> None:
        """Set all Research windows owning this physical bar."""
        self._window_models_for_bar = {int(key): value for key, value in models.items()}
        if len(self._window_models_for_bar) == 1:
            self.model_window, self.fitted_model = next(
                iter(self._window_models_for_bar.items())
            )
        elif self._window_models_for_bar:
            self.model_window = None
            self.fitted_model = None
        else:
            self.model_window = None
            self.fitted_model = None

    def take_window_evaluations(self) -> list[dict[str, Any]]:
        evaluations = self._window_evaluations
        self._window_evaluations = []
        return evaluations

    def evaluate(
        self,
        market_data: Mapping[str, Any],
    ) -> StrategyDecision:
        """
        Report an active baseline position or evaluate new-entry conditions.
        """

        if self._in_trade:
            return StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.HOLD,
                reason="S2 baseline trade active; recovery is metadata only.",
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
        self._last_bar_index = 0
        self._recovery.reset()

    def get_risk_stop_price(
        self,
        *,
        entry_price: float,
        signal: StrategySignal,
        market_data: Mapping[str, Any] | None = None,
    ) -> float | None:
        if signal is not StrategySignal.SHORT:
            return None
        return float(entry_price) + float(self.config.stop_points)

    def get_entry_fill_price(
        self,
        *,
        signal: StrategySignal,
        market_data: Mapping[str, Any],
    ) -> float | None:
        if signal is not StrategySignal.SHORT:
            return None
        return float(market_data["close"])

    def on_fill(
        self,
        *,
        market_data: Mapping[str, Any],
        position: Any,
    ) -> None:
        if position is None:
            raise RuntimeError("S2R entry fill did not create a position.")
        if not self._in_trade:
            self.start_trade(
                entry_price=float(position.entry_price),
                entry_bar=self._last_bar_index,
            )

    def on_market_data(
        self,
        market_data: Mapping[str, Any],
        position: Any,
    ) -> StrategyDecision:
        if position is None or not self._in_trade:
            raise RuntimeError("S2R lifecycle has no active trade state.")
        self._last_bar_index += 1
        # Recovery remains an analytical state stream. It must never suppress
        # or replace the validated S2 baseline stop, target, or timeout.
        self.update_trade_from_market(
            bar_index=self._last_bar_index,
            high=float(market_data["high"]),
            close=float(market_data["close"]),
        )

        stop_price = self._entry_price + float(self.config.stop_points)
        target_price = self._entry_price - (
            float(self.config.stop_points) * float(self.config.rr)
        )
        high = float(market_data["high"])
        low = float(market_data["low"])

        target_hit = low <= target_price
        stop_hit = high >= stop_price
        baseline_exit: tuple[float, str] | None = None

        if target_hit and stop_hit:
            baseline_exit = (
                stop_price,
                "S2 baseline both-hit conservative stop "
                "(NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)",
            )
        elif target_hit:
            baseline_exit = (
                target_price,
                "S2 baseline target (NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)",
            )
        elif stop_hit:
            baseline_exit = (
                stop_price,
                "S2 baseline stop (NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)",
            )
        elif self._last_bar_index >= self.config.horizon_bars:
            baseline_exit = (
                float(market_data["close"]),
                "S2 baseline timeout (NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)",
            )

        if baseline_exit is not None:
            self._pending_exit_price, reason = baseline_exit
            return StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.EXIT,
                reason=reason,
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason="S2R recovery is being tracked",
        )

    def on_exit(self) -> None:
        self.finish_trade()
        self._pending_exit_price = None

    def get_exit_fill_price(
        self,
        *,
        market_data: Mapping[str, Any],
    ) -> float | None:
        price = self._pending_exit_price
        self._pending_exit_price = None
        return price

    def set_fitted_model(
        self,
        fitted_model: S2FittedModel,
        *,
        window: int,
    ) -> None:
        self._window_models_for_bar = {}
        self.fitted_model = fitted_model
        self.model_window = int(window)
