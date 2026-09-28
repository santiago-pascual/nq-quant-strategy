from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
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
from src.models.windowed_regime import (
    ResearchHMMWindow,
    WindowedLiveHMM,
    build_research_hmm_windows,
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
    # Retained for configuration compatibility; Research 08AA uses an
    # expanding percentile against strictly prior observations.
    zscore_window: int = 30

    price_column: str = "close"


class _RankNode:
    __slots__ = ("key", "priority", "count", "size", "left", "right")

    def __init__(self, key: float, priority: float) -> None:
        self.key = key
        self.priority = priority
        self.count = 1
        self.size = 1
        self.left: _RankNode | None = None
        self.right: _RankNode | None = None


class _OrderStatisticTreap:
    def __init__(self) -> None:
        self._root: _RankNode | None = None
        self._priorities = random.Random(0)

    @staticmethod
    def _size(node: _RankNode | None) -> int:
        return node.size if node is not None else 0

    def insert(self, key: float) -> None:
        self._root = self._insert(self._root, key)

    def _insert(self, node: _RankNode | None, key: float) -> _RankNode:
        if node is None:
            return _RankNode(key, self._priorities.random())
        if key == node.key:
            node.count += 1
        elif key < node.key:
            node.left = self._insert(node.left, key)
            if node.left.priority < node.priority:
                node = self._rotate_right(node)
        else:
            node.right = self._insert(node.right, key)
            if node.right.priority < node.priority:
                node = self._rotate_left(node)
        node.size = node.count + self._size(node.left) + self._size(node.right)
        return node

    def rank_less_equal(self, key: float) -> int:
        node = self._root
        rank = 0
        while node is not None:
            if key < node.key:
                node = node.left
            else:
                rank += self._size(node.left) + node.count
                node = node.right
        return rank

    @classmethod
    def _rotate_right(cls, node: _RankNode) -> _RankNode:
        root = node.left
        assert root is not None
        node.left = root.right
        root.right = node
        node.size = node.count + cls._size(node.left) + cls._size(node.right)
        root.size = root.count + cls._size(root.left) + cls._size(root.right)
        return root

    @classmethod
    def _rotate_left(cls, node: _RankNode) -> _RankNode:
        root = node.right
        assert root is not None
        node.right = root.left
        root.left = node
        node.size = node.count + cls._size(node.left) + cls._size(node.right)
        root.size = root.count + cls._size(root.left) + cls._size(root.right)
        return root


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

    1. WINDOWED LIVE
       The current chronological research-window model is frozen at its
       pre-window training boundary and forward-filtered one bar at a time.

    2. RESEARCH WINDOW
       A fresh HMM is fitted for every explicit OOS window, using only
       observations strictly before the OOS start, then the complete OOS
       sequence is Viterbi-decoded for historical reproduction.

    Whole-OOS research Viterbi decoding and causal live filtering are
    intentionally different procedures.
    """

    def __init__(
        self,
        config: MarketContextConfig | None = None,
    ) -> None:
        self.config = config or MarketContextConfig()

        self._bars: deque[dict[str, Any]] = deque(maxlen=62)
        self._bars_seen = 0
        self._feature_rows: deque[dict[str, Any]] = deque(maxlen=500)
        self._hmm_feature_history: list[dict[str, Any]] = []
        self._volatility_rank_tree = _OrderStatisticTreap()
        self._volatility_observations = 0

        self._hmm: VolatilityRegimeModel | None = None
        self._hmm_fitted_through: pd.Timestamp | None = None
        self._research_window_definitions: tuple[ResearchHMMWindow, ...] | None = None
        self._windowed_hmm: WindowedLiveHMM | None = None

        self._research_models: dict[int, VolatilityRegimeModel] = {}
        self._s2_models: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bars_seen(self) -> int:
        return self._bars_seen

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        if not self._bars:
            return None
        return pd.Timestamp(self._bars[-1]["timestamp"])

    @property
    def hmm_fitted(self) -> bool:
        return (
            self._hmm is not None
            or (
                self._windowed_hmm is not None
                and bool(self._windowed_hmm.fitted_windows)
            )
        )

    @property
    def hmm_fitted_through(self) -> pd.Timestamp | None:
        return self._hmm_fitted_through

    @property
    def research_models(self) -> dict[int, VolatilityRegimeModel]:
        return dict(self._research_models)

    @property
    def s2_models(self) -> dict[int, Any]:
        return dict(self._s2_models)

    @property
    def windowed_hmm(self) -> WindowedLiveHMM | None:
        return self._windowed_hmm

    def configure_research_windows(
        self,
        windows: Sequence[ResearchHMMWindow],
    ) -> None:
        if self._bars_seen:
            raise RuntimeError("Research windows must be configured before replay.")
        self._windowed_hmm = WindowedLiveHMM(
            windows,
            min_train_valid=self.config.hmm_min_train_valid,
            model_factory=self._new_hmm,
        )
        self._research_window_definitions = tuple(windows)

    def prime_history(self, features: pd.DataFrame) -> None:
        """Seed bounded context and HMM training history before a replay slice."""
        if self._bars_seen:
            raise RuntimeError("Market context has already consumed bars.")
        if self._windowed_hmm is None:
            raise RuntimeError("Research windows must be configured before priming.")
        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "realized_vol_30",
            *HMM_FEATURES,
        }
        missing = required - set(features.columns)
        if missing:
            raise KeyError(f"Context warmup missing columns: {sorted(missing)}")
        history = features.copy()
        history["timestamp"] = pd.to_datetime(
            history["timestamp"], utc=True, errors="raise"
        )
        if not history["timestamp"].is_monotonic_increasing:
            raise ValueError("Context warmup timestamps must be chronological.")
        self._windowed_hmm.prime_history(history)
        self._bars.extend(
            self._normalize_bar(row)
            for row in history.tail(62).to_dict(orient="records")
        )
        self._feature_rows.extend(history.tail(500).to_dict(orient="records"))
        self._hmm_feature_history = [
            {
                "timestamp": row["timestamp"],
                **{name: row[name] for name in HMM_FEATURES},
            }
            for row in history.to_dict(orient="records")
        ]
        volatility = pd.to_numeric(
            history["realized_vol_30"], errors="coerce"
        ).to_numpy(dtype=float)
        for value in volatility:
            if np.isfinite(value):
                self._volatility_rank_tree.insert(float(value))
                self._volatility_observations += 1
        self._bars_seen = len(history)

    def _causal_volatility_percentile(self, value: Any) -> float | None:
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(current):
            return None
        percentile = (
            100.0
            * self._volatility_rank_tree.rank_less_equal(current)
            / self._volatility_observations
            if self._volatility_observations
            else None
        )
        self._volatility_rank_tree.insert(current)
        self._volatility_observations += 1
        return percentile

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._bars.clear()
        self._bars_seen = 0
        self._feature_rows.clear()
        self._hmm_feature_history.clear()
        self._volatility_rank_tree = _OrderStatisticTreap()
        self._volatility_observations = 0

        self._hmm = None
        self._hmm_fitted_through = None
        self._windowed_hmm = (
            WindowedLiveHMM(
                self._research_window_definitions,
                min_train_valid=self.config.hmm_min_train_valid,
                model_factory=self._new_hmm,
            )
            if self._research_window_definitions is not None
            else None
        )

        self._research_models.clear()
        self._s2_models.clear()

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
        self._bars_seen += 1

        frame = self._build_features(pd.DataFrame(self._bars))
        current_feature_row = frame.iloc[-1].to_dict()
        self._feature_rows.append(current_feature_row)
        self._hmm_feature_history.append(
            {
                "timestamp": current_feature_row["timestamp"],
                **{name: current_feature_row.get(name) for name in HMM_FEATURES},
            }
        )
        volatility_percentile = self._causal_volatility_percentile(
            current_feature_row.get("realized_vol_30")
        )
        context_frame = pd.DataFrame(self._feature_rows)

        context = self._build_context_from_frame(
            context_frame,
            bar,
            current_feature_row=current_feature_row,
            volatility_percentile=volatility_percentile,
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
        *,
        current_feature_row: Mapping[str, Any] | None = None,
        volatility_percentile: float | None = None,
    ) -> dict[str, Any]:
        current = frame.iloc[-1]

        context = dict(raw_bar)
        feature_values = current_feature_row or current.to_dict()

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
            if column in feature_values:
                context[column] = self._safe_float(feature_values[column])

        context.update(self._calculate_directional_features(frame))
        context["zscore"] = self._calculate_zscore(frame)

        context["vol_percentile"] = volatility_percentile

        if self._windowed_hmm is not None:
            live_features = {
                **feature_values,
                **{
                    name: context.get(name)
                    for name in (
                        "directional_pressure_30",
                        "close_location_30",
                        "normalized_momentum_30",
                    )
                },
            }
            context["hmm_state"] = self._windowed_hmm.update(
                {
                    "timestamp": live_features["timestamp"],
                    **{
                        name: live_features.get(name)
                        for name in (
                            *HMM_FEATURES,
                            "past_return_30",
                            "directional_pressure_30",
                            "close_location_30",
                            "normalized_momentum_30",
                        )
                    },
                }
            )
            context["hmm_window"] = self._active_window_number(
                pd.Timestamp(feature_values["timestamp"])
            )
        else:
            context["hmm_state"] = self._calculate_online_hmm_state(frame)

        context["market_context_ready"] = (
            context["hmm_state"] is not None
            and context["zscore"] is not None
            and context["vol_percentile"] is not None
        )

        return context

    @staticmethod
    def _calculate_directional_features(frame: pd.DataFrame) -> dict[str, float | None]:
        if len(frame) < 30 or "log_return" not in frame:
            return {
                "directional_pressure_30": None,
                "close_location_30": None,
                "normalized_momentum_30": None,
            }
        returns = pd.to_numeric(frame["log_return"].tail(30), errors="coerce")
        if returns.isna().any():
            return {
                "directional_pressure_30": None,
                "close_location_30": None,
                "normalized_momentum_30": None,
            }
        upside = float(returns.clip(lower=0).sum())
        downside = float((-returns).clip(lower=0).sum())
        total = upside + downside
        closes = pd.to_numeric(frame["close"].tail(30), errors="coerce")
        width = float(closes.max() - closes.min())
        momentum = CausalMarketContext._safe_float(frame["past_return_30"].iloc[-1])
        volatility = CausalMarketContext._safe_float(
            frame["realized_vol_30"].iloc[-1]
        )
        return {
            "directional_pressure_30": (
                (upside - downside) / total if total > 0 else None
            ),
            "close_location_30": (
                (float(closes.iloc[-1]) - float(closes.min())) / width
                if width > 0
                else None
            ),
            "normalized_momentum_30": (
                momentum / volatility
                if momentum is not None and volatility is not None and volatility > 0
                else None
            ),
        }

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

        if len(self._hmm_feature_history) < 2:
            return None

        history = pd.DataFrame(self._hmm_feature_history)
        current_timestamp = pd.Timestamp(history["timestamp"].iloc[-1])

        if self._hmm is None:
            train = history.iloc[:-1].copy()

            valid_train = self._valid_hmm_rows(train)

            if len(valid_train) < self.config.hmm_min_train_valid:
                return None

            model = self._new_hmm()

            model.fit(train)

            self._hmm = model

            self._hmm_fitted_through = current_timestamp

        current = history.iloc[[-1]].copy()

        valid_current = self._valid_hmm_rows(current)

        if valid_current.empty:
            return None

        states = self._hmm.predict_states(current)

        if states.empty:
            return None

        return int(states.iloc[-1])

    def _active_window_number(self, timestamp: pd.Timestamp) -> int | None:
        if self._windowed_hmm is None:
            return None
        for window in self._windowed_hmm.windows:
            if window.oos_start <= timestamp <= window.oos_end:
                return window.window
        return None

    def s2_model_for_window(self, window: int):
        if self._windowed_hmm is None:
            raise RuntimeError("Windowed HMM is not configured.")
        cached = self._s2_models.get(window)
        if cached is not None:
            return cached
        model = self._windowed_hmm.models.get(window)
        if model is None:
            raise RuntimeError(f"HMM window {window} has not been fitted.")
        training = self._windowed_hmm.training_frame(window)
        required = {
            "past_return_30",
            "directional_pressure_30",
            "close_location_30",
            "normalized_momentum_30",
            "realized_vol_30",
        }
        missing = required - set(training.columns)
        if missing:
            raise KeyError(
                "S2R training history missing fields: " + ", ".join(sorted(missing))
            )
        states = model.predict_states(training)
        training["hmm_state"] = states
        from src.strategies.s2r.fitting import fit_s2_model

        rows = training[
            [
                "hmm_state",
                "past_return_30",
                "directional_pressure_30",
                "close_location_30",
                "normalized_momentum_30",
                "realized_vol_30",
            ]
        ].to_dict("records")
        fitted = fit_s2_model(rows)
        self._s2_models[window] = fitted
        return fitted

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
        ).std(ddof=1)

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


def build_causal_context_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the causal context feature history used by indexed replay and warmup."""
    from src.research.direction.direction_features import (
        add_directional_pressure_features,
        add_normalized_momentum_features,
        add_range_location_features,
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    timestamp_column = (
        "timestamp ET"
        if "timestamp ET" in raw.columns
        else "timestamp"
        if "timestamp" in raw.columns
        else None
    )
    if missing or timestamp_column is None:
        if timestamp_column is None:
            missing.add("timestamp or timestamp ET")
        raise ValueError(f"Raw market data missing columns: {sorted(missing)}")

    selected_columns = [
        timestamp_column,
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    if "market_period" in raw.columns:
        selected_columns.append("market_period")
    features = raw[selected_columns].copy()
    source_timestamps = pd.to_datetime(features[timestamp_column], errors="raise")
    if source_timestamps.dt.tz is None:
        source_timestamps = source_timestamps.dt.tz_localize("UTC")
    features["timestamp"] = source_timestamps.dt.tz_convert("UTC")
    features["timestamp ET"] = source_timestamps.dt.tz_convert(
        "America/New_York"
    )
    features = features.sort_values("timestamp").reset_index(drop=True)
    features = add_return_features(features)
    features = add_volatility_features(features)
    features = add_directional_pressure_features(features)
    features = add_range_location_features(features)
    features = add_normalized_momentum_features(features)
    close = pd.to_numeric(features["close"], errors="coerce")
    features["zscore_30"] = (
        (close - close.rolling(30).mean()) / close.rolling(30).std()
    )
    return features


class PaperMarketContextEngine:
    """Indexed replay facade over the incremental causal market context."""

    def __init__(
        self,
        raw: pd.DataFrame,
        *,
        min_train_valid: int = 500,
    ) -> None:
        features = build_causal_context_features(raw)
        self.features = features

        local = features["timestamp"].dt.tz_convert("America/New_York")
        minutes = local.dt.hour * 60 + local.dt.minute
        if "market_period" in features.columns:
            rth_mask = features["market_period"].eq("RTH")
        else:
            rth_mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
        windows = build_research_hmm_windows(
            features.loc[rth_mask],
            n_windows=22,
            timestamp_column="timestamp",
            event_column="zscore_30",
        )
        self._context = CausalMarketContext(
            MarketContextConfig(hmm_min_train_valid=min_train_valid)
        )
        self._context.configure_research_windows(windows)
        self._next_index = 0

    def find_first_hmm_ready_index(self) -> int | None:
        valid = (
            self.features[HMM_FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .all(axis=1)
        )
        timestamps = self.features["timestamp"]
        for window in self._context.windowed_hmm.windows:
            training_valid = valid & (timestamps < window.oos_start)
            if int(training_valid.sum()) < self._context.config.hmm_min_train_valid:
                continue
            in_window = (
                valid
                & (timestamps >= window.oos_start)
                & (timestamps <= window.oos_end)
            )
            matches = np.flatnonzero(in_window.to_numpy())
            if len(matches):
                return int(matches[0])
        return None

    def process_bar(self, index: int) -> dict[str, Any]:
        if index < self._next_index or index >= len(self.features):
            raise ValueError("process_bar index must advance within the feature data.")
        if self._next_index == 0 and index > 0:
            self._context.prime_history(self.features.iloc[:index])
            self._next_index = index
        context: dict[str, Any] | None = None
        while self._next_index <= index:
            row = self.features.iloc[self._next_index]
            context = self._context.update(
                {
                    "timestamp": row["timestamp"],
                    "timestamp ET": row["timestamp ET"],
                    **{
                        name: row[name]
                        for name in ("open", "high", "low", "close", "volume")
                    },
                }
            )
            for name in (
                "timestamp ET",
                "zscore_30",
                "past_return_30",
                "directional_pressure_30",
                "close_location_30",
                "normalized_momentum_30",
            ):
                context[name] = row[name]
            context["zscore"] = row["zscore_30"]
            self._next_index += 1
        if context is None:
            raise RuntimeError("No context was generated.")
        return context
