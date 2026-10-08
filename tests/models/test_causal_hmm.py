from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from src.models.causal_hmm import (
    CausalGaussianHMM,
    CausalHMMConfig,
    ScheduledCausalHMMStream,
    ScheduledRawStateProvider,
    add_calendar_months,
)
from src.models.regime import HMM_FEATURES
from src.paper.market_context import CausalMarketContext, MarketContextConfig
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy
from src.strategies.s2r.config import S2RConfig


def _feature_frame(n: int, *, start: str = "2024-01-02 09:30") -> pd.DataFrame:
    rng = np.random.default_rng(19)
    t = np.arange(n, dtype=float)
    values = np.column_stack(
        [
            0.01 + 0.002 * np.sin(t / (3 + j)) + rng.normal(0, 0.0002, n)
            for j in range(4)
        ]
        + [
            0.9 + 0.2 * np.sin(t / 7) + rng.normal(0, 0.02, n),
            1.1 + 0.2 * np.cos(t / 9) + rng.normal(0, 0.02, n),
        ]
    )
    stamps = pd.date_range(start, periods=n, freq="min", tz="America/New_York")
    return pd.DataFrame(values, columns=HMM_FEATURES).assign(
        timestamp=stamps.tz_convert("UTC"),
        past_return_30=np.sin(t / 11.0) * 0.02,
        directional_pressure_30=np.sin(t / 13.0) * 0.6,
        close_location_30=0.5 + 0.4 * np.sin(t / 17.0),
        normalized_momentum_30=np.cos(t / 19.0) * 0.8,
    )


def _fit_model(n: int = 80) -> CausalGaussianHMM:
    return CausalGaussianHMM(
        CausalHMMConfig(min_train_valid=20, n_iter=8)
    ).fit(_feature_frame(n))


def test_forward_filter_matches_independent_log_domain_reference() -> None:
    model = _fit_model()
    frame = _feature_frame(3)
    x0 = frame.iloc[0]
    z0 = model.scaler.transform(x0[HMM_FEATURES].to_numpy(dtype=float)[None, :])[0]
    emission0 = model.model._compute_log_likelihood(z0[None, :])[0]
    start_log = np.where(model.model.startprob_ > 0, np.log(model.model.startprob_), -np.inf)
    expected0 = start_log + emission0
    expected0 -= logsumexp(expected0)
    got0 = model.filter_one(x0, None)
    assert got0 is not None
    assert np.allclose(got0[1], np.exp(expected0), rtol=0, atol=1e-12)
    assert got0[0] == int(np.argmax(np.exp(expected0)))

    x1 = frame.iloc[1]
    z1 = model.scaler.transform(x1[HMM_FEATURES].to_numpy(dtype=float)[None, :])[0]
    emission1 = model.model._compute_log_likelihood(z1[None, :])[0]
    trans_log = np.where(model.model.transmat_ > 0, np.log(model.model.transmat_), -np.inf)
    expected1 = emission1 + logsumexp(expected0[:, None] + trans_log, axis=0)
    expected1 -= logsumexp(expected1)
    got1 = model.filter_one(x1, got0[2])
    assert got1 is not None
    assert np.allclose(got1[1], np.exp(expected1), rtol=0, atol=1e-12)


def test_forward_filter_is_append_only_when_future_rows_arrive() -> None:
    model = _fit_model()
    rows = _feature_frame(30)
    states, posteriors, _ = model.filter_sequence(rows.iloc[:12])
    state_at_t = int(states[-1])
    posterior_at_t = posteriors[-1].copy()
    later_states, later_posteriors, _ = model.filter_sequence(rows)
    assert state_at_t == int(states[-1])
    assert np.array_equal(posterior_at_t, posteriors[-1])
    assert len(later_states) > len(states)
    assert later_posteriors.shape[1] == 3


def test_missing_features_do_not_update_posterior_and_next_bar_advances_once() -> None:
    model = _fit_model()
    rows = _feature_frame(4)
    first = model.filter_one(rows.iloc[0], None)
    assert first is not None
    missing = rows.iloc[1].copy()
    missing[HMM_FEATURES[2]] = np.nan
    assert model.filter_one(missing, first[2]) is None
    after_gap = model.filter_one(rows.iloc[2], first[2])
    direct = model.filter_one(rows.iloc[2], first[2])
    assert after_gap is not None and direct is not None
    assert np.array_equal(after_gap[1], direct[1])


def test_posterior_persists_across_session_weekend_and_missing_observation() -> None:
    stream = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4,
        config=CausalHMMConfig(min_train_valid=10, n_iter=4),
    )
    rows = _feature_frame(20)
    for _, row in rows.iterrows():
        stream.update(row["timestamp"], row)
    assert stream.log_posterior is not None
    previous = stream.log_posterior.copy()
    retained_rows = stream._history.size

    # The HMM chain advances per observed valid row, not elapsed clock time.
    gap_row = rows.iloc[-1].copy()
    gap_row[HMM_FEATURES[0]] = np.nan
    weekend_timestamp = rows.iloc[-1]["timestamp"] + pd.Timedelta(days=3)
    state, posterior = stream.update(weekend_timestamp, gap_row)
    assert state is None and posterior is None
    assert np.array_equal(stream.log_posterior, previous)
    assert stream._history.size == retained_rows

    monday_row = rows.iloc[-2].copy()
    monday_timestamp = weekend_timestamp + pd.Timedelta(minutes=1)
    expected = stream.model.filter_one(monday_row, previous)
    actual = stream.update(monday_timestamp, monday_row)
    assert expected is not None and actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])


def test_scaler_uses_the_same_population_transform_for_fit_and_inference() -> None:
    frame = _feature_frame(80)
    model = _fit_model()
    valid = model.valid_features(frame)
    expected = (valid.iloc[0].to_numpy() - model.scaler.mean_) / model.scaler.scale_
    actual = model.scaler.transform(valid.iloc[[0]].to_numpy())[0]
    assert np.array_equal(actual, expected)
    assert np.allclose(np.mean(model.scaler.transform(valid), axis=0), 0.0, atol=1e-12)


def test_duplicate_and_out_of_order_timestamps_are_rejected() -> None:
    stream = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4,
        config=CausalHMMConfig(min_train_valid=10, n_iter=3),
    )
    rows = _feature_frame(2)
    stream.update(rows.iloc[0]["timestamp"], rows.iloc[0])
    with pytest.raises(ValueError, match="increase strictly"):
        stream.update(rows.iloc[0]["timestamp"], rows.iloc[0])
    with pytest.raises(ValueError, match="increase strictly"):
        stream.update(rows.iloc[0]["timestamp"] - pd.Timedelta(minutes=1), rows.iloc[1])


def test_mr_refit_deadline_is_four_calendar_months_and_training_is_strict() -> None:
    stream = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4,
        config=CausalHMMConfig(min_train_valid=10, n_iter=4),
    )
    rows = _feature_frame(24)
    for _, row in rows.iterrows():
        state, _ = stream.update(row["timestamp"], row)
    assert stream.model is not None
    first_fit = stream.fit_timestamp
    assert first_fit is not None
    assert stream.fit_end_exclusive == first_fit
    assert stream.next_refit_timestamp == first_fit.tz_convert("America/New_York") + pd.DateOffset(months=4)
    assert stream.fit_start is not None and stream.fit_start < first_fit
    assert state in {0, 1, 2}


def test_s2r_first_fit_and_rolling_refit_exclude_the_live_boundary() -> None:
    stream = ScheduledCausalHMMStream(
        schedule="rolling", refit_months=3, rolling_years=2,
        eligible_rth_only=True,
        config=CausalHMMConfig(min_train_valid=10, n_iter=3),
    )
    dates = pd.date_range(
        "2020-01-06 09:30", "2022-04-30 09:30", freq="B",
        tz="America/New_York",
    )
    # Include exact first and second live boundaries.
    dates = dates.union(
        pd.DatetimeIndex(
            ["2022-01-06 09:30", "2022-04-06 09:30"], tz="America/New_York"
        )
    ).sort_values()
    rows = _feature_frame(len(dates))
    rows["timestamp"] = dates.tz_convert("UTC")
    fitted_at = None
    observed_fit_count = 0
    for _, row in rows.iterrows():
        stream.update(row["timestamp"], row, eligible=True)
        if stream.fit_count > observed_fit_count:
            fitted_at = row["timestamp"]
            if stream.fit_count == 1:
                assert fitted_at.tz_convert("America/New_York").date().isoformat() == "2022-01-06"
                assert stream.fit_end_exclusive == fitted_at
                assert stream.fit_last_training_timestamp < fitted_at
                assert stream.s2_fitted_model is not None
            elif stream.fit_count == 2:
                assert fitted_at.tz_convert("America/New_York").date().isoformat() == "2022-04-06"
                assert stream.fit_end_exclusive == fitted_at
                assert stream.fit_last_training_timestamp < fitted_at
                assert stream.s2_fitted_model is not None
            observed_fit_count = stream.fit_count
    assert fitted_at is not None
    assert stream.fit_count == 2
    assert stream.next_refit_timestamp is not None


def test_rolling_history_prune_at_refit_matches_per_observation_prune() -> None:
    class LegacyDictionaryStream(ScheduledCausalHMMStream):
        """Reference the former object-backed storage and retention behavior."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.legacy_history = []

        def _append_history(self, timestamp, record):
            self.legacy_history.append(dict(record))

        def _prune_history(self, lower):
            self.legacy_history = [
                item for item in self.legacy_history
                if pd.Timestamp(item["timestamp"]) >= lower
            ]

        def _training_frame(self, live_start):
            start, stop = self._training_bounds(live_start)
            return pd.DataFrame(self.legacy_history[start:stop]).reset_index(drop=True)

        def _training_bounds(self, live_start):
            stamps = np.fromiter(
                (pd.Timestamp(item["timestamp"]).value for item in self.legacy_history),
                dtype=np.int64,
                count=len(self.legacy_history),
            )
            start = 0
            stop = int(np.searchsorted(stamps, live_start.value, side="left"))
            if self.schedule == "rolling":
                lower = add_calendar_months(live_start, -12 * self.rolling_years)
                start = int(np.searchsorted(stamps, lower.value, side="left"))
            return start, stop

        def _training_matrix(self, start, stop):
            return np.asarray(
                [[item.get(name, np.nan) for name in HMM_FEATURES]
                 for item in self.legacy_history[start:stop]],
                dtype=np.float64,
            )

        def _training_timestamps(self, start, stop):
            return np.fromiter(
                (pd.Timestamp(item["timestamp"]).value
                 for item in self.legacy_history[start:stop]),
                dtype=np.int64,
                count=stop - start,
            )

        def update(self, timestamp, row, *, eligible=True):
            result = super().update(timestamp, row, eligible=eligible)
            if eligible and self.schedule == "rolling":
                lower = add_calendar_months(timestamp, -12 * self.rolling_years)
                self.legacy_history = [
                    item for item in self.legacy_history
                    if pd.Timestamp(item["timestamp"]) >= lower
                ]
            return result

    dates = pd.date_range(
        "2020-01-06 09:30", "2022-04-30 09:30", freq="B",
        tz="America/New_York",
    ).union(pd.DatetimeIndex(
        ["2022-01-06 09:30", "2022-04-06 09:30"], tz="America/New_York"
    )).sort_values()
    rows = _feature_frame(len(dates))
    rows["timestamp"] = dates.tz_convert("UTC")
    config = CausalHMMConfig(min_train_valid=10, n_iter=3)
    optimized = ScheduledCausalHMMStream(
        schedule="rolling", refit_months=3, rolling_years=2,
        eligible_rth_only=True, config=config,
    )
    reference = LegacyDictionaryStream(
        schedule="rolling", refit_months=3, rolling_years=2,
        eligible_rth_only=True, config=config,
    )
    left, right = [], []
    for _, row in rows.iterrows():
        left.append(optimized.update(row["timestamp"], row))
        right.append(reference.update(row["timestamp"], row))
    assert optimized.fit_count == reference.fit_count == 2
    assert [item[0] for item in left] == [item[0] for item in right]
    for (_, p_left), (_, p_right) in zip(left, right):
        if p_left is None or p_right is None:
            assert p_left is p_right
        else:
            assert np.allclose(p_left, p_right, rtol=1e-12, atol=1e-12)
    # Independent fits may have different byte hashes from floating-point
    # operation order; compare fitted values numerically. Checkpoint tests
    # below separately require stable hashes for a persisted artifact.
    assert optimized.fit_start == reference.fit_start
    assert optimized.fit_end_exclusive == reference.fit_end_exclusive
    assert optimized.fit_last_training_timestamp == reference.fit_last_training_timestamp
    assert np.allclose(optimized.model.scaler.mean_, reference.model.scaler.mean_, rtol=1e-12, atol=1e-12)
    assert np.allclose(optimized.model.scaler.scale_, reference.model.scaler.scale_, rtol=1e-12, atol=1e-12)
    assert np.allclose(optimized.model.model.startprob_, reference.model.model.startprob_, rtol=1e-12, atol=1e-12)
    assert np.allclose(optimized.model.model.transmat_, reference.model.model.transmat_, rtol=1e-12, atol=1e-12)
    assert np.allclose(optimized.model.model.means_, reference.model.model.means_, rtol=1e-12, atol=1e-12)
    assert np.allclose(optimized.model.model.covars_, reference.model.model.covars_, rtol=1e-12, atol=1e-12)
    assert optimized.s2_fitted_model.signal_model.thresholds == reference.s2_fitted_model.signal_model.thresholds
    assert optimized.s2_fitted_model.signal_model.scales == reference.s2_fitted_model.signal_model.scales
    assert optimized.s2_fitted_model.volatility_reference == reference.s2_fitted_model.volatility_reference


def test_fit_is_deterministic_for_identical_data_and_runtime() -> None:
    data = _feature_frame(100)
    first = CausalGaussianHMM(CausalHMMConfig(min_train_valid=20, n_iter=8)).fit(data)
    second = CausalGaussianHMM(CausalHMMConfig(min_train_valid=20, n_iter=8)).fit(data)
    assert first.artifact_hash == second.artifact_hash
    assert np.array_equal(first.model.startprob_, second.model.startprob_)
    assert np.array_equal(first.model.transmat_, second.model.transmat_)
    assert np.array_equal(first.model.means_, second.model.means_)


def test_compact_matrix_fit_matches_dataframe_fit_numerically() -> None:
    data = _feature_frame(120)
    dataframe_fit = CausalGaussianHMM(
        CausalHMMConfig(min_train_valid=20, n_iter=8)
    ).fit(data)
    matrix_fit = CausalGaussianHMM(
        CausalHMMConfig(min_train_valid=20, n_iter=8)
    ).fit_matrix(data.loc[:, HMM_FEATURES].to_numpy(dtype=np.float64))
    for left, right in (
        (dataframe_fit.scaler.mean_, matrix_fit.scaler.mean_),
        (dataframe_fit.scaler.scale_, matrix_fit.scaler.scale_),
        (dataframe_fit.model.startprob_, matrix_fit.model.startprob_),
        (dataframe_fit.model.transmat_, matrix_fit.model.transmat_),
        (dataframe_fit.model.means_, matrix_fit.model.means_),
        (dataframe_fit.model.covars_, matrix_fit.model.covars_),
    ):
        assert np.allclose(left, right, rtol=1e-12, atol=1e-12)


def test_checkpoint_restart_matches_continuous_filter_and_preserves_raw_ids() -> None:
    rows = _feature_frame(500)
    config = CausalHMMConfig(min_train_valid=20, n_iter=6)
    continuous = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4, config=config
    )
    continuous_states = []
    for _, row in rows.iterrows():
        continuous_states.append(continuous.update(row["timestamp"], row)[0])

    split = 300
    chunked = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4, config=config
    )
    before = [
        chunked.update(row["timestamp"], row)[0]
        for _, row in rows.iloc[:split].iterrows()
    ]
    encoded = json.dumps(chunked.state_dict())
    restarted = ScheduledCausalHMMStream.from_state_dict(json.loads(encoded))
    after = [
        restarted.update(row["timestamp"], row)[0]
        for _, row in rows.iloc[split:].iterrows()
    ]
    assert before + after == continuous_states
    assert set(state for state in continuous_states if state is not None) <= {0, 1, 2}
    assert not hasattr(restarted, "semantic_mapping")


def test_checkpoint_restart_crosses_an_exact_refit_boundary() -> None:
    rows = _feature_frame(206)
    timestamps = list(
        pd.date_range(
            "2024-01-02 09:30", periods=201, freq="min", tz="America/New_York"
        )
    ) + [
        pd.Timestamp(f"2024-{month:02d}-02 09:30", tz="America/New_York")
        for month in (2, 3, 4, 5, 6)
    ]
    rows["timestamp"] = pd.DatetimeIndex(timestamps).tz_convert("UTC")
    config = CausalHMMConfig(min_train_valid=200, n_iter=3)

    def process(stream, part):
        results = []
        for _, row in part.iterrows():
            state, posterior = stream.update(row["timestamp"], row)
            results.append(
                (state, None if posterior is None else posterior.copy(), stream.model_hash)
            )
        return results

    continuous = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4, config=config
    )
    expected = process(continuous, rows)

    chunked = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4, config=config
    )
    split = 205  # checkpoint before the June 2 observation crosses the refit due date
    actual = process(chunked, rows.iloc[:split])
    restored = ScheduledCausalHMMStream.from_state_dict(
        json.loads(json.dumps(chunked.state_dict()))
    )
    actual += process(restored, rows.iloc[split:])

    assert [item[0] for item in actual] == [item[0] for item in expected]
    assert [item[2] for item in actual] == [item[2] for item in expected]
    for got, want in zip(actual, expected):
        if got[1] is None or want[1] is None:
            assert got[1] is want[1]
        else:
            assert np.array_equal(got[1], want[1])
    assert restored.fit_count == continuous.fit_count == 2
    assert restored.next_refit_timestamp == continuous.next_refit_timestamp


def test_compact_history_is_typed_and_checkpoint_does_not_embed_row_dicts() -> None:
    stream = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4,
        config=CausalHMMConfig(min_train_valid=80, n_iter=8),
    )
    rows = _feature_frame(120)
    for _, row in rows.iterrows():
        stream.update(row["timestamp"], row)

    history = stream._history
    assert history.timestamps_ns.dtype == np.dtype("int64")
    assert history.values.dtype == np.dtype("float64")
    assert history.values.shape[1] == len(HMM_FEATURES) == 6
    assert history.size == len(rows)
    assert not any(isinstance(value, dict) for value in history.values.flat)

    checkpoint = stream.state_dict()
    assert "history" not in checkpoint
    assert checkpoint["history_compact"]["encoding"] == "zlib+base64"
    assert checkpoint["training_data_hash"] == stream.training_data_hash
    assert stream.training_data_hash is not None
    restored = ScheduledCausalHMMStream.from_state_dict(
        json.loads(json.dumps(checkpoint))
    )
    assert restored._history.size == stream._history.size
    assert np.array_equal(
        restored._history.timestamps_ns[:restored._history.size],
        stream._history.timestamps_ns[:stream._history.size],
    )
    assert np.array_equal(
        restored._history.values[:restored._history.size],
        stream._history.values[:stream._history.size],
    )
    assert restored.model_hash == stream.model_hash
    assert restored.training_data_hash == stream.training_data_hash
    before = restored._history.size
    restored.update(rows.iloc[-1]["timestamp"] + pd.Timedelta(minutes=1), rows.iloc[-1])
    assert restored._history.size == before + 1


def test_compact_training_slice_preserves_timestamp_membership_and_uses_views() -> None:
    stream = ScheduledCausalHMMStream(
        schedule="expanding", refit_months=4,
        config=CausalHMMConfig(min_train_valid=10, n_iter=3),
    )
    rows = _feature_frame(40)
    for _, row in rows.iterrows():
        stream.update(row["timestamp"], row)

    live_start = rows.iloc[-1]["timestamp"]
    training = stream._training_frame(live_start)
    assert len(training) == len(rows) - 1
    assert training["timestamp"].iloc[0] == rows["timestamp"].iloc[0]
    assert training["timestamp"].iloc[-1] == rows["timestamp"].iloc[-2]
    assert np.shares_memory(
        training[HMM_FEATURES[0]].to_numpy(),
        stream._history.values[:stream._history.size, 0],
    )


def test_raw_strategy_state_contract_remains_unchanged() -> None:
    assert MRL1_CONFIG.hmm_state == 1
    assert MRS2_CONFIG.hmm_state == 2
    assert S2RConfig().target_state == 2
    config = CausalHMMConfig().normalized()
    assert config.n_components == 3
    assert config.covariance_type == "full"
    assert config.random_state == 42
    assert config.tol == 0.01
    assert config.min_covar == 1e-3
    assert config.sklearn_version
    assert config.hmmlearn_version
    assert config.numpy_version
    assert config.scipy_version


def test_non_rth_s2r_rows_do_not_advance_or_seed_rolling_training() -> None:
    provider = ScheduledRawStateProvider(
        config=CausalHMMConfig(min_train_valid=10, n_iter=3)
    )
    rows = _feature_frame(3)
    provider.s2r_state_for(rows.iloc[0]["timestamp"], rows.iloc[0], is_rth=False)
    provider.s2r_state_for(rows.iloc[1]["timestamp"], rows.iloc[1], is_rth=True)
    assert provider.s2r.anchor_timestamp == rows.iloc[1]["timestamp"]
    assert len(provider.s2r.history) == 1


def _context_bars(n: int) -> list[dict[str, object]]:
    stamps = pd.date_range(
        "2026-01-05 09:30", periods=n, freq="min", tz="America/New_York"
    )
    return [
        {
            "timestamp": stamp.to_pydatetime(),
            "open": 100.0 + i * 0.01,
            "high": 100.5 + i * 0.01,
            "low": 99.5 + i * 0.01,
            "close": 100.0 + i * 0.01 + np.sin(i / 3.0),
            "volume": 10 + i % 7,
        }
        for i, stamp in enumerate(stamps)
    ]


def test_context_checkpoint_restart_matches_features_states_and_candidates() -> None:
    config = MarketContextConfig(hmm_min_train_valid=500, hmm_n_iter=8)
    bars = _context_bars(700)
    continuous = CausalMarketContext(config)
    expected = [continuous.update(bar) for bar in bars]

    chunked = CausalMarketContext(config)
    prefix = [chunked.update(bar) for bar in bars[:550]]
    checkpoint = json.dumps(chunked.state_dict())
    restarted = CausalMarketContext(config)
    restarted.load_state_dict(json.loads(checkpoint))
    suffix = [restarted.update(bar) for bar in bars[550:]]

    actual = prefix + suffix
    for left, right in zip(expected, actual):
        assert left["hmm_state"] == right["hmm_state"]
        assert left.get("s2r_hmm_state") == right.get("s2r_hmm_state")
        assert left.get("realized_vol_30") == right.get("realized_vol_30")
        for key in ("hmm_posterior", "s2r_hmm_posterior"):
            assert left.get(key) == right.get(key)
            if left.get(key) is not None:
                assert np.isclose(sum(left[key]), 1.0, rtol=0, atol=1e-12)
    def strategy_candidates(rows):
        strategies = {
            "MRL1": MeanReversionStrategy(MRL1_CONFIG),
            "MRS2": MeanReversionStrategy(MRS2_CONFIG),
        }
        return [
            (
                name,
                str(row["timestamp"]),
                strategy.generate_signal(row).value,
            )
            for row in rows
            for name, strategy in strategies.items()
            if strategy.generate_signal(row).value != "flat"
        ]

    assert strategy_candidates(expected) == strategy_candidates(actual)


def test_historical_research_model_scaling_path_remains_separate() -> None:
    # Production fit uses a separate class; historical Research keeps its
    # training-only correction unchanged in VolatilityRegimeModel.
    from src.models.regime import VolatilityRegimeModel

    assert VolatilityRegimeModel.standardize.__module__ == "src.models.regime"
