from __future__ import annotations

import json
import math
from pathlib import Path
import os
import uuid

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.paper.bootstrap_artifacts import (
    BootstrapProgressWriter,
    build_replay_causal_features,
    load_or_build_causal_feature_cache,
)
from src.paper.market_context import CausalMarketContext, MarketContextConfig


def _bars(count: int = 720) -> pd.DataFrame:
    index = np.arange(count, dtype=float)
    returns = np.random.default_rng(42).normal(0.0, 2.0, count)
    close = 20_000.0 + np.cumsum(returns) + np.sin(index / 11.0) * 1.5
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01T13:30:00Z", periods=count, freq="min"),
        "open": close - 0.25,
        "high": close + 0.75,
        "low": close - 0.75,
        "close": close,
        "volume": 100.0 + (index % 17),
    })


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent


def _remove_cache_files(*infos):
    for info in infos:
        if not info:
            continue
        data_path = Path(info["path"])
        data_path.unlink(missing_ok=True)
        data_path.with_suffix(".json").unlink(missing_ok=True)


def _without_timing_fields(value):
    if isinstance(value, dict):
        return {
            key: _without_timing_fields(item)
            for key, item in value.items()
            if not key.endswith("_seconds") and "duration_seconds" not in key
        }
    if isinstance(value, list):
        return [_without_timing_fields(item) for item in value]
    return value


def _first_difference(left, right, path="state"):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return (path, list(left.keys()), list(right.keys()))
        for key in left:
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return (path + ".length", len(left), len(right))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            found = _first_difference(left_item, right_item, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return None
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return None
    return None if left == right else (path, repr(left), repr(right))


def test_feature_cache_round_trip_corruption_and_source_identity():
    raw = _bars()
    directory = _cache_dir()
    first_info = cached_info = changed_info = identified_info = contract_changed_info = None
    try:
        first, first_info = load_or_build_causal_feature_cache(raw, directory)
        uncached = build_replay_causal_features(raw)
        cached, cached_info = load_or_build_causal_feature_cache(raw, directory)
        assert first_info["status"] in {"miss_built", "miss_rebuilt"}
        assert cached_info["status"] == "hit"
        assert_frame_equal(first, uncached, check_exact=True)
        assert_frame_equal(first, cached, check_exact=True)

        cache_path = Path(first_info["path"])
        cache_path.write_bytes(cache_path.read_bytes()[:-8] + b"corrupt!")
        rebuilt, rebuilt_info = load_or_build_causal_feature_cache(raw, directory)
        assert rebuilt_info["status"] == "miss_rebuilt"
        assert "checksum-invalid" in rebuilt_info["rejected_cache_reason"]
        assert_frame_equal(first, rebuilt, check_exact=True)

        changed = raw.copy()
        changed.loc[10, "close"] += 0.25
        _changed, changed_info = load_or_build_causal_feature_cache(changed, directory)
        assert changed_info["cache_key"] != first_info["cache_key"]

        identified = raw.assign(symbol="MNQ.v.0", instrument_id=42005282)
        _identified_features, identified_info = load_or_build_causal_feature_cache(identified, directory)
        changed_contract = identified.assign(instrument_id=42004800)
        _contract_features, contract_changed_info = load_or_build_causal_feature_cache(
            changed_contract, directory
        )
        assert identified_info["cache_key"] != contract_changed_info["cache_key"]
    finally:
        _remove_cache_files(first_info, changed_info, identified_info, contract_changed_info)


def test_feature_cache_prefix_is_independent_of_appended_future_rows():
    raw = _bars(190)
    cutoff = raw["timestamp"].iloc[150]
    prefix = raw.loc[raw["timestamp"] < cutoff].copy()
    extended = raw.loc[raw["timestamp"] >= cutoff].copy()
    directory = _cache_dir()
    prefix_info = extended_info = None
    try:
        prefix_features, prefix_info = load_or_build_causal_feature_cache(prefix, directory)
        # The cache API receives only the causal prefix, even when the caller
        # possesses later bars. Those later bars are not inputs to the build.
        appended_prefix = pd.concat([prefix, extended], ignore_index=True).iloc[:len(prefix)].copy()
        extended_features, extended_info = load_or_build_causal_feature_cache(appended_prefix, directory)
        assert prefix_info["cache_key"] == extended_info["cache_key"]
        assert_frame_equal(prefix_features, extended_features, check_exact=True)
        assert pd.to_datetime(prefix_features["timestamp"], utc=True).max() < cutoff
    finally:
        _remove_cache_files(prefix_info, extended_info)


def test_bootstrap_context_state_round_trip_matches_uninterrupted_continuation():
    raw = _bars()
    directory = _cache_dir()
    features_info = None
    try:
        features, features_info = load_or_build_causal_feature_cache(raw, directory)
        config = MarketContextConfig(hmm_n_iter=20, hmm_min_train_valid=120)

        uninterrupted = CausalMarketContext(config)
        uninterrupted.bootstrap_causal_history(raw, precomputed_features=features)

        split = 400
        partial = CausalMarketContext(config)
        partial.bootstrap_causal_history(
            raw.iloc[:split].copy(),
            precomputed_features=features.iloc[:split].copy(),
        )
        # Match the repository's current JSON checkpoint encoding, which uses
        # Python's JSON NaN extension for missing rolling feature values.
        saved_state = json.loads(json.dumps(partial.state_dict()))
        resumed = CausalMarketContext(config)
        resumed.load_state_dict(saved_state)
        for raw_row, feature_row in zip(
            raw.iloc[split:].to_dict("records"),
            features.iloc[split:].to_dict("records"),
        ):
            resumed.update_with_precomputed_features(raw_row, feature_row)

        difference = _first_difference(
            _without_timing_fields(uninterrupted.state_dict()),
            _without_timing_fields(resumed.state_dict()),
        )
        assert difference is None, difference
        assert uninterrupted.bars_seen == resumed.bars_seen == len(raw)
        assert uninterrupted.last_timestamp == resumed.last_timestamp
    finally:
        _remove_cache_files(features_info)


def test_causal_bootstrap_callback_reports_rows_and_refit_boundaries():
    raw = _bars(720)
    directory = _cache_dir()
    features_info = None
    try:
        features, features_info = load_or_build_causal_feature_cache(raw, directory)
        events = []
        context = CausalMarketContext(
            MarketContextConfig(hmm_n_iter=10, hmm_min_train_valid=100)
        )
        context.bootstrap_causal_history(
            raw, precomputed_features=features, progress_callback=events.append
        )
        assert events[-1]["rows_processed"] == len(raw)
        assert events[-1]["total_rows"] == len(raw)
        assert events[-1]["last_timestamp"] == raw["timestamp"].iloc[-1]
        assert any(event["hmm_refit_status"] == "MR_refit_in_progress" for event in events)
        assert any(event["hmm_refit_status"] == "MR_refit_completed" for event in events)
        assert all(event["total_rows"] == len(raw) for event in events)
    finally:
        _remove_cache_files(features_info)


def test_bootstrap_state_rejects_incompatible_context_configuration():
    saved = CausalMarketContext(MarketContextConfig(hmm_n_iter=5)).state_dict()
    with pytest.raises(ValueError, match="configuration differs"):
        CausalMarketContext(MarketContextConfig()).load_state_dict(saved)


def test_bootstrap_state_rejects_corrupt_version_and_hmm_schema():
    saved = CausalMarketContext(MarketContextConfig()).state_dict()
    corrupt_version = json.loads(json.dumps(saved))
    corrupt_version["version"] = 999
    with pytest.raises(ValueError, match="Unsupported market-context checkpoint version"):
        CausalMarketContext(MarketContextConfig()).load_state_dict(corrupt_version)

    corrupt_schema = json.loads(json.dumps(saved))
    corrupt_schema["hmm_provider"]["feature_schema"] = ["future_feature"]
    with pytest.raises(ValueError, match="Unsupported raw-state HMM provider checkpoint"):
        CausalMarketContext(MarketContextConfig()).load_state_dict(corrupt_schema)


def test_bootstrap_progress_writes_atomic_stage_and_refit_status():
    path = Path(__file__).resolve().parent / f".bootstrap_progress_{os.getpid()}_{uuid.uuid4().hex}.json"
    try:
        writer = BootstrapProgressWriter(path, flush_interval_seconds=60)
        writer.stage("causal_hmm_bootstrap", total_rows=4)
        writer.update(
            rows_processed=2,
            total_rows=4,
            last_timestamp=pd.Timestamp("2026-07-01T13:31:00Z"),
            hmm_refit_status="MR_refit_in_progress",
            refit_timestamp=pd.Timestamp("2026-07-01T13:32:00Z"),
            force=True,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["current_stage"] == "causal_hmm_bootstrap"
        assert payload["historical_rows_processed"] == 2
        assert payload["historical_rows_total"] == 4
        assert payload["last_processed_timestamp_utc"] == "2026-07-01T13:31:00+00:00"
        assert payload["current_hmm_refit_status"] == "MR_refit_in_progress"
        assert payload["current_refit_timestamp_utc"] == "2026-07-01T13:32:00+00:00"
        assert not list(path.parent.glob(path.name + ".*.tmp"))
        writer.finish()
        assert json.loads(path.read_text(encoding="utf-8"))["completed"] is True
    finally:
        path.unlink(missing_ok=True)
        for temporary in path.parent.glob(path.name + ".*.tmp"):
            temporary.unlink(missing_ok=True)
