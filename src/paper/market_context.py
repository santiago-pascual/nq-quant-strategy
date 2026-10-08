from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
from src.models.hmm_provider import HMMStateProvider
from src.models.causal_hmm import (
    CausalHMMConfig,
    S2R_FEATURES,
    ScheduledRawStateProvider,
)
from src.models.windowed_regime import (
    ResearchHMMWindow,
    WindowedLiveHMM,
    build_research_hmm_windows,
)


PRECOMPUTED_CONTEXT_FEATURES = tuple(dict.fromkeys((
    "timestamp",
    "open", "high", "low", "close", "volume",
    "log_return", "return", "long_return", "short_return",
    "past_return_1", "past_return_3", "past_return_5", "past_return_10",
    "past_return_15", "past_return_30",
    "realized_vol_5", "realized_vol_15", "realized_vol_30", "realized_vol_60",
    "vol_ratio_5_30", "vol_ratio_5_60", "variance_5", "variance_30",
    "variance_60", "variance_ratio_5_30", "variance_ratio_5_60",
    *S2R_FEATURES,
)))


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


class CausalOnlineHMMProvider:
    """Live provider with scheduled forward filters and unchanged raw IDs."""

    TEST_ONLY = False

    def __init__(self, context: "CausalMarketContext") -> None:
        self._context = context

    def state_for(
        self,
        timestamp: pd.Timestamp,
        features: pd.DataFrame | Mapping[str, Any] | None,
    ) -> int | None:
        del timestamp
        if features is None:
            return None
        return self._context._calculate_online_hmm_state(features)

    def s2r_state_for(
        self, timestamp: pd.Timestamp, row: Mapping[str, Any], *, is_rth: bool
    ) -> int | None:
        return self._context._calculate_online_s2r_state(
            timestamp, row, is_rth=is_rth
        )

    @property
    def mr_model_hash(self) -> str | None:
        return self._context._raw_hmm_provider.mr.model_hash

    @property
    def mr_model_version(self) -> int:
        return self._context._raw_hmm_provider.mr.fit_count

    @property
    def mr_next_refit_timestamp(self) -> pd.Timestamp | None:
        return self._context._raw_hmm_provider.mr.next_refit_timestamp

    @property
    def mr_posterior(self) -> tuple[float, ...] | None:
        log_posterior = self._context._raw_hmm_provider.mr.log_posterior
        return (
            tuple(np.exp(log_posterior).tolist())
            if log_posterior is not None else None
        )

    @property
    def current_s2r_fitted_model(self):
        return self._context._raw_hmm_provider.s2r.s2_fitted_model

    @property
    def s2r_model_hash(self) -> str | None:
        return self._context._raw_hmm_provider.s2r.model_hash

    @property
    def s2r_model_version(self) -> int:
        return self._context._raw_hmm_provider.s2r.fit_count

    @property
    def s2r_posterior(self) -> tuple[float, ...] | None:
        log_posterior = self._context._raw_hmm_provider.s2r.log_posterior
        return (
            tuple(np.exp(log_posterior).tolist())
            if log_posterior is not None else None
        )

    def state_dict(self) -> dict[str, Any]:
        return self._context._raw_hmm_provider.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._context._raw_hmm_provider = ScheduledRawStateProvider.from_state_dict(state)

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

    The default provider is CausalOnlineHMMProvider. The Research Replay has
    its own TEST-ONLY FrozenResearchHMMProvider behind a separate context
    adapter; strategy code receives the same ``hmm_state`` field in either
    mode and does not select or inspect the provider.
    """

    def __init__(
        self,
        config: MarketContextConfig | None = None,
        *,
        hmm_provider: HMMStateProvider | None = None,
    ) -> None:
        self.config = config or MarketContextConfig()

        self._bars: deque[dict[str, Any]] = deque(maxlen=62)
        # Mean Reversion's validated z-score samples RTH closes only. Keep
        # this separate from the raw rolling feature window, which includes ETH.
        self._rth_closes: deque[float] = deque(maxlen=self.config.zscore_window)
        self._bars_seen = 0
        self._feature_rows: deque[dict[str, Any]] = deque(maxlen=500)
        # Only the causal volatility observations are needed to reconstruct
        # the expanding prior-volatility rank tree at checkpoint restore.
        # HMM training rows live in the two scheduled streams.
        self._hmm_feature_history: list[float] = []
        self._volatility_rank_tree = _OrderStatisticTreap()
        self._volatility_observations = 0

        self._hmm: VolatilityRegimeModel | None = None
        self._hmm_fitted_through: pd.Timestamp | None = None
        self._raw_hmm_provider = self._new_raw_hmm_provider()
        self._research_window_definitions: tuple[ResearchHMMWindow, ...] | None = None
        self._windowed_hmm: WindowedLiveHMM | None = None

        self._research_models: dict[int, VolatilityRegimeModel] = {}
        self._s2_models: dict[int, Any] = {}
        self.hmm_provider = hmm_provider or CausalOnlineHMMProvider(self)

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
            or self._raw_hmm_provider.mr.fitted
            or (
                self._windowed_hmm is not None
                and bool(self._windowed_hmm.fitted_windows)
            )
        )

    @property
    def hmm_fitted_through(self) -> pd.Timestamp | None:
        if self._raw_hmm_provider.mr.fit_timestamp is not None:
            return self._raw_hmm_provider.mr.fit_timestamp
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

    @property
    def causal_hmm_refit_events(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "mr": tuple(self._raw_hmm_provider.mr.refit_events),
            "s2r": tuple(self._raw_hmm_provider.s2r.refit_events),
        }

    def configure_research_windows(
        self,
        windows: Sequence[ResearchHMMWindow],
    ) -> None:
        """Configure the legacy windowed HMM path for explicit Research tests."""
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
        for row in history.to_dict(orient="records"):
            if not self._is_research_rth(row["timestamp"]):
                continue
            value = self._safe_float(row.get("realized_vol_30"))
            if value is not None:
                self._volatility_rank_tree.insert(value)
                self._volatility_observations += 1
                self._hmm_feature_history.append(value)
            self._rth_closes.append(float(row["close"]))
        self._bars_seen = len(history)

    def bootstrap_causal_history(
        self,
        raw_bars: pd.DataFrame,
        *,
        precomputed_features: pd.DataFrame | None = None,
    ) -> pd.Timestamp:
        """Prime production causal state from ordered historical raw OHLCV.

        This path builds the same right-aligned causal feature columns once,
        then advances both HMM streams observation-by-observation. It omits
        Paper engine events and per-bar context materialization before the
        requested strategy interval.
        """
        if self._bars_seen or self._raw_hmm_provider.mr.last_timestamp is not None:
            raise RuntimeError("Causal history bootstrap requires an empty context.")
        features = (
            precomputed_features
            if precomputed_features is not None
            else build_causal_context_features(raw_bars)
        )
        if precomputed_features is not None:
            timestamp_column = "timestamp" if "timestamp" in raw_bars else "timestamp ET"
            if timestamp_column not in raw_bars or len(features) != len(raw_bars):
                raise ValueError("Precomputed causal features do not match raw history.")
            raw_stamps = pd.to_datetime(raw_bars[timestamp_column], utc=True, errors="raise")
            feature_stamps = pd.to_datetime(features["timestamp"], utc=True, errors="raise")
            if not raw_stamps.reset_index(drop=True).equals(
                feature_stamps.reset_index(drop=True)
            ):
                raise ValueError("Precomputed causal feature timestamps differ from raw history.")
        if features.empty:
            raise ValueError("Causal history bootstrap received no bars.")
        if not features["timestamp"].is_monotonic_increasing:
            raise ValueError("Causal bootstrap timestamps must be chronological.")
        if features["timestamp"].duplicated().any():
            raise ValueError("Causal bootstrap timestamps must be unique.")

        # Prepare compact numeric matrices once. Iteration below still advances
        # every observed row in order, so neither forward chain skips a step.
        names = tuple(dict.fromkeys((*HMM_FEATURES, *S2R_FEATURES)))
        feature_index = {name: index for index, name in enumerate(names)}
        matrix = features.loc[:, names].to_numpy(dtype=np.float64, copy=True)
        # The incremental directional helper requires a complete 30-return
        # sample before it emits any of its three S2R values. The batch close
        # location feature alone becomes finite one close earlier, so preserve
        # the incremental warmup contract explicitly.
        directional_ready = (
            features["log_return"].rolling(30).count().to_numpy() == 30
        )
        matrix[~directional_ready, feature_index["close_location_30"]] = np.nan
        raw_matrix = features.loc[:, ("open", "high", "low", "close", "volume")].to_numpy(
            dtype=np.float64, copy=False
        )
        timestamps = features["timestamp"].array
        local = features["timestamp ET"]
        minute = local.dt.hour.to_numpy() * 60 + local.dt.minute.to_numpy()
        is_rth = (minute >= 9 * 60 + 30) & (minute < 17 * 60)
        mr_width = len(HMM_FEATURES)
        valid_hmm = np.isfinite(matrix[:, :mr_width]).all(axis=1)

        self._raw_hmm_provider = self._new_raw_hmm_provider()
        self._raw_hmm_provider.mr._history.reserve(int(valid_hmm.sum()))
        self._raw_hmm_provider.s2r._history.reserve(
            int((valid_hmm & is_rth).sum())
        )
        for index, timestamp in enumerate(timestamps):
            stamp = pd.Timestamp(timestamp)
            row = dict(zip(names, matrix[index]))
            self._raw_hmm_provider.mr.update(stamp, row)
            self._raw_hmm_provider.s2r.update(
                stamp, row, eligible=bool(is_rth[index])
            )
            if is_rth[index]:
                value = self._safe_float(
                    matrix[index, feature_index["realized_vol_30"]]
                )
                if value is not None:
                    self._volatility_rank_tree.insert(value)
                    self._volatility_observations += 1
                    self._hmm_feature_history.append(value)
                self._rth_closes.append(float(raw_matrix[index, 3]))

        self._bars.extend(
            self._normalize_bar(row)
            for row in features.loc[
                :, ["timestamp", "open", "high", "low", "close", "volume"]
            ].tail(62).to_dict(orient="records")
        )
        self._feature_rows.extend(features.tail(500).to_dict(orient="records"))
        self._bars_seen = len(features)
        return pd.Timestamp(features["timestamp"].iloc[-1])

    def update_with_precomputed_features(
        self,
        market_data: Mapping[str, Any],
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume one bar using its causal, already-computed feature row."""
        bar = self._normalize_bar(market_data)
        stamp = pd.Timestamp(features["timestamp"])
        if stamp.tzinfo is None:
            raise ValueError("Precomputed feature timestamps must be timezone-aware.")
        stamp = stamp.tz_convert("UTC")
        if stamp != bar["timestamp"]:
            raise ValueError("Precomputed features must match the current bar timestamp.")
        feature_row = dict(features)
        feature_row["timestamp"] = stamp
        self._bars.append(bar)
        self._bars_seen += 1
        self._feature_rows.append(feature_row)

        is_rth = self._is_research_rth(stamp)
        if is_rth:
            self._rth_closes.append(float(bar["close"]))
            volatility = self._safe_float(feature_row.get("realized_vol_30"))
            if volatility is not None:
                self._hmm_feature_history.append(volatility)
            percentile = self._causal_volatility_percentile(volatility)
        else:
            percentile = None
        return self._build_context_from_frame(
            None,
            bar,
            current_feature_row=feature_row,
            volatility_percentile=percentile,
        )

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
        self._rth_closes.clear()
        self._bars_seen = 0
        self._feature_rows.clear()
        self._hmm_feature_history.clear()
        self._volatility_rank_tree = _OrderStatisticTreap()
        self._volatility_observations = 0

        self._hmm = None
        self._hmm_fitted_through = None
        self._raw_hmm_provider = self._new_raw_hmm_provider()
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
        is_rth = self._is_research_rth(bar["timestamp"])
        if is_rth:
            self._rth_closes.append(float(bar["close"]))
            volatility_value = self._safe_float(
                current_feature_row.get("realized_vol_30")
            )
            if volatility_value is not None:
                self._hmm_feature_history.append(volatility_value)
            volatility_percentile = self._causal_volatility_percentile(
                current_feature_row.get("realized_vol_30")
            )
        else:
            volatility_percentile = None
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
        context = dict(raw_bar)
        if current_feature_row is None:
            if frame is None or frame.empty:
                raise ValueError("A context frame or current feature row is required.")
            feature_values = frame.iloc[-1].to_dict()
        else:
            feature_values = current_feature_row

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

        directional_names = (
            "directional_pressure_30", "close_location_30", "normalized_momentum_30"
        )
        if all(name in feature_values for name in directional_names):
            context.update({
                name: self._safe_float(feature_values.get(name))
                for name in directional_names
            })
        else:
            context.update(self._calculate_directional_features(frame))
        context["zscore"] = self._calculate_zscore(frame)
        is_rth = self._is_research_rth(raw_bar["timestamp"])
        context["market_period"] = "RTH" if is_rth else "ETH"
        if not is_rth:
            context["zscore"] = None

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
            context["hmm_state"] = self.hmm_provider.state_for(
                pd.Timestamp(feature_values["timestamp"]), feature_values
            )
            context["hmm_posterior"] = getattr(
                self.hmm_provider, "mr_posterior", None
            )
            s2r_row = {
                **feature_values,
                **{
                    name: context.get(name)
                    for name in S2R_FEATURES
                },
            }
            s2r_state_for = getattr(self.hmm_provider, "s2r_state_for", None)
            context["s2r_hmm_state"] = (
                s2r_state_for(
                    pd.Timestamp(feature_values["timestamp"]),
                    s2r_row,
                    is_rth=is_rth,
                )
                if s2r_state_for is not None
                else None
            )
            context["s2r_hmm_posterior"] = getattr(
                self.hmm_provider, "s2r_posterior", None
            )
            context["s2r_model_hash"] = getattr(
                self.hmm_provider, "s2r_model_hash", None
            )
            context["s2r_model_version"] = getattr(
                self.hmm_provider, "s2r_model_version", None
            )
            context["hmm_model_hash"] = getattr(
                self.hmm_provider, "mr_model_hash", None
            )
            context["hmm_model_version"] = getattr(
                self.hmm_provider, "mr_model_version", None
            )
            next_mr_refit = getattr(
                self.hmm_provider, "mr_next_refit_timestamp", None
            )
            context["hmm_next_refit_timestamp"] = (
                next_mr_refit.isoformat() if next_mr_refit is not None else None
            )

        context["market_context_ready"] = (
            context["hmm_state"] is not None
            and context["zscore"] is not None
            and context["vol_percentile"] is not None
        )

        return context

    @staticmethod
    def _is_research_rth(timestamp: Any) -> bool:
        """Research session gate: 09:30 inclusive to 17:00 exclusive, New York."""
        local = pd.Timestamp(timestamp)
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert("America/New_York")
        minute = local.hour * 60 + local.minute
        return 9 * 60 + 30 <= minute < 17 * 60

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
        frame: pd.DataFrame | Mapping[str, Any],
    ) -> int | None:
        """
        Online path.

        The first model fit uses only observations strictly before
        the current bar. Once fitted, the model is frozen.

        This is intentionally separate from fit_research_window(),
        which exists to reproduce Research 08b exactly.
        """

        if isinstance(frame, pd.DataFrame):
            if frame.empty:
                return None
            current = frame.iloc[-1].to_dict()
        else:
            current = frame
        timestamp = pd.Timestamp(current["timestamp"])
        state, _posterior = self._raw_hmm_provider.mr.update(timestamp, current)
        return state

    def _calculate_online_s2r_state(
        self,
        timestamp: pd.Timestamp,
        row: Mapping[str, Any],
        *,
        is_rth: bool,
    ) -> int | None:
        state, _posterior = self._raw_hmm_provider.s2r.update(
            timestamp, row, eligible=is_rth
        )
        return state

    def state_dict(self) -> dict[str, Any]:
        """Serializable restart state for causal features and both HMM streams."""
        from src.models.causal_hmm import _json_value

        return {
            "version": 2,
            "context_config": asdict(self.config),
            "bars": _json_value(list(self._bars)),
            "feature_rows": _json_value(list(self._feature_rows)),
            "hmm_feature_history": _json_value(self._hmm_feature_history),
            "rth_closes": list(self._rth_closes),
            "bars_seen": self._bars_seen,
            "hmm_provider": self._raw_hmm_provider.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", 0)) != 2:
            raise ValueError("Unsupported market-context checkpoint version.")
        if dict(state.get("context_config", {})) != asdict(self.config):
            raise ValueError("Market-context checkpoint configuration differs.")
        if self._bars_seen:
            raise RuntimeError("Restore context state before processing bars.")
        self._bars.extend(self._normalize_bar(row) for row in state["bars"])
        self._feature_rows.extend(state["feature_rows"])
        self._hmm_feature_history = []
        self._rth_closes.extend(float(value) for value in state["rth_closes"])
        self._bars_seen = int(state["bars_seen"])
        self._raw_hmm_provider = ScheduledRawStateProvider.from_state_dict(
            state["hmm_provider"]
        )
        for row in state["hmm_feature_history"]:
            # Version-2 checkpoints written before compact bootstrap stored
            # timestamp/feature dictionaries; continue reading those states.
            if isinstance(row, Mapping):
                if not self._is_research_rth(row["timestamp"]):
                    continue
                value = self._safe_float(row.get("realized_vol_30"))
            else:
                value = self._safe_float(row)
            if value is not None:
                self._volatility_rank_tree.insert(value)
                self._volatility_observations += 1
                self._hmm_feature_history.append(value)

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist an exact causal feature/HMM continuation checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(self.state_dict(), separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(target)

    def restore_checkpoint(self, path: str | Path) -> None:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_state_dict(state)

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

        if len(self._rth_closes) < window:
            return None

        close = np.asarray(self._rth_closes, dtype=float)
        close_value = close[-1]
        mean_value = float(close.mean())
        std_value = float(close.std(ddof=1))

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

    def _new_raw_hmm_provider(self) -> ScheduledRawStateProvider:
        return ScheduledRawStateProvider(
            config=CausalHMMConfig(
                n_components=self.config.hmm_n_states,
                random_state=self.config.hmm_random_state,
                n_iter=self.config.hmm_n_iter,
                min_train_valid=self.config.hmm_min_train_valid,
            )
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
    local_minutes = (
        features["timestamp ET"].dt.hour * 60
        + features["timestamp ET"].dt.minute
    )
    if "market_period" not in features:
        features["market_period"] = np.where(
            local_minutes.between(9 * 60 + 30, 17 * 60 - 1), "RTH", "ETH"
        )
    features = add_return_features(features)
    features = add_volatility_features(features)
    features = add_directional_pressure_features(features)
    features = add_range_location_features(features)
    features = add_normalized_momentum_features(features)
    rth_close = pd.to_numeric(
        features["close"].where(features["market_period"].eq("RTH")),
        errors="coerce",
    ).dropna()
    rth_zscore = (rth_close - rth_close.rolling(30).mean()) / rth_close.rolling(30).std()
    features["zscore_30"] = rth_zscore.reindex(features.index)
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
            rth_mask = (minutes >= 9 * 60 + 30) & (minutes < 17 * 60)
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
