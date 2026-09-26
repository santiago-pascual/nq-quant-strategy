from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)
from src.models.regime import (
    HMM_FEATURES,
    VolatilityRegimeModel,
)


@dataclass(frozen=True)
class MarketContextConfig:
    """
    Configuration for causal market-context construction.
    """

    hmm_n_states: int = 3
    hmm_random_state: int = 42
    hmm_n_iter: int = 200
    hmm_min_train_valid: int = 500

    volatility_percentile_window: int = 500
    zscore_window: int = 30

    price_column: str = "close"


class CausalMarketContext:
    """
    Stateful feature/context builder for the paper engine.

    IMPORTANT
    ---------
    This class does not use any generated research cache.

    It calculates:
        OHLCV
          -> returns
          -> volatility
          -> z-score
          -> volatility percentile
          -> HMM context

    HMM handling has two modes:

    1. ONLINE
       A frozen fitted model is used for current-bar inference.

    2. RESEARCH WINDOW
       A fresh HMM is fitted for every explicit OOS window, using
       only observations strictly before the OOS start.

    The second mode exists specifically to reproduce Research 08b.
    """

    def __init__(
        self,
        config: MarketContextConfig | None = None,
    ) -> None:
        self.config = config or MarketContextConfig()

        self._bars: list[dict[str, Any]] = []

        self._hmm: VolatilityRegimeModel | None = None
        self._hmm_fitted_through: pd.Timestamp | None = None

        self._research_models: dict[int, VolatilityRegimeModel] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bars_seen(self) -> int:
        return len(self._bars)

    @property
    def hmm_fitted(self) -> bool:
        return self._hmm is not None

    @property
    def hmm_fitted_through(self) -> pd.Timestamp | None:
        return self._hmm_fitted_through

    @property
    def research_models(self) -> dict[int, VolatilityRegimeModel]:
        return dict(self._research_models)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._bars.clear()

        self._hmm = None
        self._hmm_fitted_through = None

        self._research_models.clear()

    # ------------------------------------------------------------------
    # Normal live/online update
    # ------------------------------------------------------------------

    def update(
        self,
        market_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Add one observed bar and calculate its causal market context.

        No future bar is available to this method.
        """

        bar = self._normalize_bar(market_data)

        self._bars.append(bar)

        frame = self._build_features(pd.DataFrame(self._bars))

        context = self._build_context_from_frame(
            frame,
            bar,
        )

        return context

    # ------------------------------------------------------------------
    # Research-window HMM
    # ------------------------------------------------------------------

    def fit_research_window(
        self,
        features: pd.DataFrame,
        *,
        window: int,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Reproduce the HMM construction used by Research 08b.

        Training:
            canonical_timestamp < oos_start

        OOS:
            oos_start <= timestamp <= oos_end

        A completely fresh HMM is fitted for this window.

        This method intentionally mirrors the historical Research 08b
        methodology rather than the strictly-online inference path.
        """

        frame = features.copy()

        frame = self._normalize_feature_frame(frame)

        train = frame.loc[frame["canonical_timestamp"] < pd.Timestamp(oos_start)].copy()

        oos = frame.loc[
            (frame["canonical_timestamp"] >= pd.Timestamp(oos_start))
            & (frame["canonical_timestamp"] <= pd.Timestamp(oos_end))
        ].copy()

        valid_train = self._valid_hmm_rows(train)

        valid_oos = self._valid_hmm_rows(oos)

        if len(valid_train) < self.config.hmm_min_train_valid:
            raise ValueError(
                f"Window {window}: insufficient causal HMM training data: "
                f"{len(valid_train)} < "
                f"{self.config.hmm_min_train_valid}"
            )

        if valid_oos.empty:
            raise ValueError(f"Window {window}: no valid HMM observations in OOS.")

        model = self._new_hmm()

        # This deliberately passes the full training dataframe.
        # VolatilityRegimeModel.prepare_data() performs the exact
        # feature selection / NaN removal used by the project model.
        model.fit(train)

        # This deliberately reproduces Research 08b:
        #
        # model.predict_states(oos)
        #
        # We do NOT replace this with a different inference method here,
        # because this class is the benchmark-reproduction path.
        states = model.predict_states(oos)

        if states.empty:
            raise RuntimeError(f"Window {window}: HMM produced no states.")

        result = pd.DataFrame(
            {
                "timestamp": oos.loc[
                    states.index,
                    "canonical_timestamp",
                ].to_numpy(),
                "window": int(window),
                "hmm_state": states.to_numpy(dtype=np.int8),
            }
        )

        result = result.sort_values("timestamp").reset_index(drop=True)

        self._research_models[int(window)] = model

        return result

    # ------------------------------------------------------------------
    # Multiple research windows
    # ------------------------------------------------------------------

    def fit_research_windows(
        self,
        features: pd.DataFrame,
        windows: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit one independent HMM per OOS window.

        Required columns in `windows`:

            window
            oos_start
            oos_end
        """

        required = {
            "window",
            "oos_start",
            "oos_end",
        }

        missing = required - set(windows.columns)

        if missing:
            raise KeyError(
                "Research-window table missing columns: " + ", ".join(sorted(missing))
            )

        results: list[pd.DataFrame] = []

        ordered_windows = (
            windows[
                [
                    "window",
                    "oos_start",
                    "oos_end",
                ]
            ]
            .drop_duplicates()
            .sort_values("window")
            .reset_index(drop=True)
        )

        for row in ordered_windows.itertuples(index=False):
            result = self.fit_research_window(
                features,
                window=int(row.window),
                oos_start=pd.Timestamp(row.oos_start),
                oos_end=pd.Timestamp(row.oos_end),
            )

            results.append(result)

        if not results:
            raise ValueError("No research windows were supplied.")

        combined = pd.concat(
            results,
            ignore_index=True,
        )

        combined["timestamp"] = pd.to_datetime(
            combined["timestamp"],
            utc=True,
        )

        combined["window"] = combined["window"].astype(np.int64)

        combined["hmm_state"] = combined["hmm_state"].astype(np.int8)

        combined = combined.sort_values(
            [
                "window",
                "timestamp",
            ]
        ).reset_index(drop=True)

        return combined

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = frame.copy()

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
        )

        frame = frame.sort_values("timestamp").reset_index(drop=True)

        frame = add_return_features(frame)
        frame = add_volatility_features(frame)

        return frame

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_context_from_frame(
        self,
        frame: pd.DataFrame,
        raw_bar: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = frame.iloc[-1]

        context = dict(raw_bar)

        feature_columns = (
            "log_return",
            "return",
            "long_return",
            "short_return",
            "past_return_1",
            "past_return_3",
            "past_return_5",
            "past_return_10",
            "past_return_15",
            "past_return_30",
            "realized_vol_5",
            "realized_vol_15",
            "realized_vol_30",
            "realized_vol_60",
            "vol_ratio_5_30",
            "vol_ratio_5_60",
            "variance_5",
            "variance_30",
            "variance_60",
            "variance_ratio_5_30",
            "variance_ratio_5_60",
        )

        for column in feature_columns:
            if column in frame.columns:
                context[column] = self._safe_float(current[column])

        context["zscore"] = self._calculate_zscore(frame)

        context["vol_percentile"] = self._calculate_volatility_percentile(frame)

        context["hmm_state"] = self._calculate_online_hmm_state(frame)

        context["market_context_ready"] = (
            context["hmm_state"] is not None
            and context["zscore"] is not None
            and context["vol_percentile"] is not None
        )

        return context

    # ------------------------------------------------------------------
    # Online HMM
    # ------------------------------------------------------------------

    def _calculate_online_hmm_state(
        self,
        frame: pd.DataFrame,
    ) -> int | None:
        """
        Online path.

        The first model fit uses only observations strictly before
        the current bar. Once fitted, the model is frozen.

        This is intentionally separate from fit_research_window(),
        which exists to reproduce Research 08b exactly.
        """

        if len(frame) < 2:
            return None

        current_timestamp = pd.Timestamp(frame["timestamp"].iloc[-1])

        if self._hmm is None:
            train = frame.iloc[:-1].copy()

            valid_train = self._valid_hmm_rows(train)

            if len(valid_train) < self.config.hmm_min_train_valid:
                return None

            model = self._new_hmm()

            model.fit(train)

            self._hmm = model

            self._hmm_fitted_through = current_timestamp

        current = frame.iloc[[-1]].copy()

        valid_current = self._valid_hmm_rows(current)

        if valid_current.empty:
            return None

        states = self._hmm.predict_states(current)

        if states.empty:
            return None

        return int(states.iloc[-1])

    # ------------------------------------------------------------------
    # Z-score
    # ------------------------------------------------------------------

    def _calculate_zscore(
        self,
        frame: pd.DataFrame,
    ) -> float | None:
        window = self.config.zscore_window

        if len(frame) < window:
            return None

        close = pd.to_numeric(
            frame[self.config.price_column],
            errors="coerce",
        )

        rolling_mean = close.rolling(
            window=window,
            min_periods=window,
        ).mean()

        rolling_std = close.rolling(
            window=window,
            min_periods=window,
        ).std(ddof=0)

        close_value = close.iloc[-1]
        mean_value = rolling_mean.iloc[-1]
        std_value = rolling_std.iloc[-1]

        if not all(
            np.isfinite(value)
            for value in (
                close_value,
                mean_value,
                std_value,
            )
        ):
            return None

        if std_value <= 0:
            return None

        return float((close_value - mean_value) / std_value)

    # ------------------------------------------------------------------
    # Volatility percentile
    # ------------------------------------------------------------------

    def _calculate_volatility_percentile(
        self,
        frame: pd.DataFrame,
    ) -> float | None:
        if "realized_vol_30" not in frame.columns:
            return None

        series = (
            pd.to_numeric(
                frame["realized_vol_30"],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if series.empty:
            return None

        window = self.config.volatility_percentile_window

        if len(series) > window:
            series = series.iloc[-window:]

        current = float(series.iloc[-1])

        if not np.isfinite(current):
            return None

        values = np.sort(series.to_numpy(dtype=float))

        if len(values) == 1:
            return 0.0

        rank = np.searchsorted(
            values,
            current,
            side="right",
        )

        return float(rank / len(values) * 100.0)

    # ------------------------------------------------------------------
    # HMM helpers
    # ------------------------------------------------------------------

    def _new_hmm(self) -> VolatilityRegimeModel:
        return VolatilityRegimeModel(
            n_states=self.config.hmm_n_states,
            random_state=self.config.hmm_random_state,
            n_iter=self.config.hmm_n_iter,
        )

    @staticmethod
    def _valid_hmm_rows(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        return (
            frame[HMM_FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

    # ------------------------------------------------------------------
    # Feature-frame normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_feature_frame(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = frame.copy()

        if "canonical_timestamp" not in frame.columns:
            if "timestamp" in frame.columns:
                frame["canonical_timestamp"] = pd.to_datetime(
                    frame["timestamp"],
                    errors="raise",
                    utc=True,
                )
            elif "timestamp ET" in frame.columns:
                frame["canonical_timestamp"] = pd.to_datetime(
                    frame["timestamp ET"],
                    errors="raise",
                    utc=True,
                )
            else:
                raise KeyError("Feature dataframe has no timestamp column.")

        else:
            frame["canonical_timestamp"] = pd.to_datetime(
                frame["canonical_timestamp"],
                errors="raise",
                utc=True,
            )

        frame = frame.sort_values("canonical_timestamp").reset_index(drop=True)

        duplicate_count = int(frame["canonical_timestamp"].duplicated().sum())

        if duplicate_count:
            raise ValueError(
                f"Feature dataframe contains {duplicate_count:,} duplicate timestamps."
            )

        missing = [feature for feature in HMM_FEATURES if feature not in frame.columns]

        if missing:
            raise KeyError(
                "Feature dataframe missing HMM features: " + ", ".join(missing)
            )

        return frame

    # ------------------------------------------------------------------
    # Bar normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_bar(
        market_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )

        missing = [field for field in required if field not in market_data]

        if missing:
            raise KeyError("Missing market-data fields: " + ", ".join(missing))

        timestamp = pd.Timestamp(market_data["timestamp"])

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        return {
            **dict(market_data),
            "timestamp": timestamp,
            "open": float(market_data["open"]),
            "high": float(market_data["high"]),
            "low": float(market_data["low"]),
            "close": float(market_data["close"]),
            "volume": float(market_data["volume"]),
        }

    # ------------------------------------------------------------------
    # Safe numeric conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not np.isfinite(value):
            return None

        return value
