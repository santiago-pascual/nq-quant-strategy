from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
import json
import math
from pathlib import Path
import shutil
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.paper.bootstrap_artifacts import build_replay_causal_features
from src.paper.causal_bootstrap import (
    CausalBootstrapCheckpointStore,
    CausalBootstrapRunner,
    _numerical_thread_identity,
    activation_payload_errors,
    load_context_seed_for_new_account,
)
from src.paper.market_context import CausalMarketContext
from src.risk.policy import XFA_50K_PRODUCTION_POLICY


NY = ZoneInfo("America/New_York")
INITIAL_EQUITY = 50_000.0


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()
                if not key.endswith("_seconds") and key != "written_at_utc"}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return "__NAN__"
    return value


def _first_difference(left, right, path="state"):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left:
            if key not in right:
                return f"{path}.{key}: missing on right"
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _without_hashes(value):
    if isinstance(value, dict):
        return {key: _without_hashes(item) for key, item in value.items()
                if "hash" not in key}
    if isinstance(value, list):
        return [_without_hashes(item) for item in value]
    return value


def _history() -> tuple[pd.DataFrame, pd.Timestamp]:
    # Bounded deterministic fixture with >2 calendar years of RTH observations
    # and enough row density for the unmodified 500-row minimum in both streams.
    dates = pd.bdate_range("2019-05-06", "2021-12-31")
    rows = []
    ordinal = 0
    for date in dates:
        start = pd.Timestamp(datetime.combine(date.date(), time(9, 30), NY)).tz_convert("UTC")
        for minute in range(3):
            close = 15_000.0 + ordinal * 0.006 + 2.2 * np.sin(ordinal / 13.0) + 0.8 * np.sin(ordinal / 4.7)
            rows.append({
                "timestamp": start + pd.Timedelta(minutes=minute),
                "symbol": "MNQ.v.0", "instrument_id": 42005282,
                "open": close - 0.25, "high": close + 0.75,
                "low": close - 0.75, "close": close,
                "volume": float(100 + (ordinal % 71)),
            })
            ordinal += 1
    raw = pd.DataFrame(rows)
    last_local = pd.Timestamp(raw.timestamp.iloc[-1]).tz_convert(NY).date()
    next_day = pd.bdate_range(last_local + pd.Timedelta(days=1), periods=1)[0].date()
    activation = pd.Timestamp(datetime.combine(next_day, time(9, 30), NY)).tz_convert("UTC")
    return raw, activation


def _coverage(_raw):
    return {"verified": True, "missing_expected_minutes": 0,
            "unknown_coverage": 0, "certificate_readiness": "FULLY_CERTIFIED",
            "calendar_coverage_certificate": {
                "readiness": "FULLY_CERTIFIED",
                "calendar_snapshots": [{"identity": "test", "review_status": "REVIEWED_TEST"}],
            },
            "source": "test-only deterministic fixture"}


@pytest.fixture(scope="module")
def bootstrap_test_root():
    root = Path(__file__).resolve().parent / f".tmp_causal_bootstrap_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def warmed_artifact(bootstrap_test_root):
    raw, activation = _history()
    features = build_replay_causal_features(raw)
    output = bootstrap_test_root / "warmup.json"
    checkpoint = CausalBootstrapCheckpointStore(output)
    runner = CausalBootstrapRunner(
        context=CausalMarketContext(), checkpoint_path=output,
        activation_timestamp=activation,
        history_origin_timestamp=raw.timestamp.iloc[0],
        feature_cache_key="fixture-causal-feature-cache-v1",
        initial_equity=INITIAL_EQUITY,
        risk_configuration=asdict(XFA_50K_PRODUCTION_POLICY),
        coverage_validator=_coverage,
        chunk_rows=257,
    )
    payload = runner.run(raw, features)
    return {"raw": raw, "features": features, "activation": activation,
            "path": output, "identity": payload["identity"], "payload": payload}


def test_resumable_causal_bootstrap_matches_uninterrupted_hmm_and_context(bootstrap_test_root):
    raw, activation = _history()
    features = build_replay_causal_features(raw)
    origin = raw.timestamp.iloc[0]
    risk = asdict(XFA_50K_PRODUCTION_POLICY)

    full_path = bootstrap_test_root / "full.json"
    full = CausalBootstrapRunner(
        context=CausalMarketContext(), checkpoint_path=full_path,
        activation_timestamp=activation, history_origin_timestamp=origin,
        feature_cache_key="same-feature-cache", initial_equity=INITIAL_EQUITY,
        risk_configuration=risk, coverage_validator=_coverage,
        chunk_rows=len(raw),
    ).run(raw, features)

    resumed_path = bootstrap_test_root / "resumed.json"

    class SimulatedPowerLoss(RuntimeError):
        pass

    def interrupt_after_durable_chunk(event):
        if event.get("checkpoint_committed") and event["rows_processed"] >= 514:
            raise SimulatedPowerLoss("simulated termination after atomic checkpoint")

    interrupted_runner = CausalBootstrapRunner(
        context=CausalMarketContext(), checkpoint_path=resumed_path,
        activation_timestamp=activation, history_origin_timestamp=origin,
        feature_cache_key="same-feature-cache", initial_equity=INITIAL_EQUITY,
        risk_configuration=risk, coverage_validator=_coverage,
        chunk_rows=257,
    )
    with pytest.raises(SimulatedPowerLoss):
        interrupted_runner.run(raw, features, progress_callback=interrupt_after_durable_chunk)

    resumed_context = CausalMarketContext()
    resumed = CausalBootstrapRunner(
        context=resumed_context, checkpoint_path=resumed_path,
        activation_timestamp=activation, history_origin_timestamp=origin,
        feature_cache_key="same-feature-cache", initial_equity=INITIAL_EQUITY,
        risk_configuration=risk, coverage_validator=_coverage,
        chunk_rows=257,
    ).run(raw, features, resume=True)

    assert full["ready_for_activation"] and resumed["ready_for_activation"]
    assert full["rows_processed"] == resumed["rows_processed"] == len(raw)
    assert resumed["activation_timestamp_utc"] == activation.isoformat()
    assert full["account_seed"]["positions"] == resumed["account_seed"]["positions"] == []
    assert full["account_seed"]["orders"] == resumed["account_seed"]["orders"] == []
    assert full["account_seed"]["trades"] == resumed["account_seed"]["trades"] == []
    assert full["strategy_state_seed"]["no_historical_strategy_execution"] is True
    assert set(full["strategy_state_seed"]["strategies"]) == {"MRL1", "MRS2", "S2R", "ORB"}
    assert full["strategy_state_seed"] == resumed["strategy_state_seed"]
    assert activation_payload_errors(full, activation_timestamp=activation,
                                     expected_identity=full["identity"]) == []
    assert full["context"]["hmm_provider"]["mr"]["fit_count"] >= 1
    assert full["context"]["hmm_provider"]["s2r"]["fit_count"] >= 1
    for name in ("mr", "s2r"):
        full_stream = full["context"]["hmm_provider"][name]
        resumed_stream = resumed["context"]["hmm_provider"][name]
        assert full_stream["training_data_hash"] == resumed_stream["training_data_hash"], name
        assert full_stream["model_identity_hash"] == resumed_stream["model_identity_hash"], (
            name, _first_difference(full_stream, resumed_stream),
            _first_difference(_without_hashes(full_stream.get("model")),
                              _without_hashes(resumed_stream.get("model"))),
        )
        assert full_stream["last_emitted_state"] == resumed_stream["last_emitted_state"], name
    assert _clean(full["context"]) == _clean(resumed["context"]), _first_difference(
        _clean(full["context"]), _clean(resumed["context"])
    )
    for name in ("mr", "s2r"):
        assert _clean(full["context"]["hmm_provider"][name]["refit_events"]) == _clean(
            resumed["context"]["hmm_provider"][name]["refit_events"]
        )
        assert full["context"]["hmm_provider"][name]["next_refit_timestamp"] == resumed["context"]["hmm_provider"][name]["next_refit_timestamp"]


def test_bootstrap_fails_closed_on_future_rows_missing_coverage_and_wrong_origin(bootstrap_test_root):
    raw, activation = _history()
    features = build_replay_causal_features(raw)
    params = dict(
        activation_timestamp=activation,
        history_origin_timestamp=raw.timestamp.iloc[0],
        feature_cache_key="fixture-cache",
        initial_equity=INITIAL_EQUITY,
        risk_configuration=asdict(XFA_50K_PRODUCTION_POLICY),
        coverage_validator=_coverage,
    )
    future_raw = pd.concat([raw, pd.DataFrame([{
        **raw.iloc[-1].to_dict(), "timestamp": activation,
    }])], ignore_index=True)
    future_features = build_replay_causal_features(future_raw)
    with pytest.raises(ValueError, match="activation-time or future"):
        CausalBootstrapRunner(
            context=CausalMarketContext(), checkpoint_path=bootstrap_test_root / "future.json", **params
        ).run(future_raw, future_features)

    with pytest.raises(ValueError, match="coverage is not fully verified"):
        CausalBootstrapRunner(
            context=CausalMarketContext(), checkpoint_path=bootstrap_test_root / "gap.json",
            **{**params, "coverage_validator": lambda _: {
                "verified": False, "missing_expected_minutes": 3,
                "unknown_coverage": 1,
            }},
        ).run(raw, features)

    wrong_origin = CausalBootstrapRunner(
        context=CausalMarketContext(), checkpoint_path=bootstrap_test_root / "origin.json",
        **{**params, "history_origin_timestamp": raw.timestamp.iloc[1]},
    )
    with pytest.raises(ValueError, match="configured historical origin"):
        wrong_origin.run(raw, features)


def test_bootstrap_accepts_only_explicit_scoped_paper_coverage_decision(bootstrap_test_root):
    raw, activation = _history()
    features = build_replay_causal_features(raw)
    paper_policy = {
        "policy": "paper_research_quality_accepted",
        "decision": "ACCEPTED_FOR_INTERNAL_SIMULATED_PAPER_ONLY",
        "scope": "internal_simulated_paper_only_no_live_execution",
        "acceptance_artifact_sha256": "a" * 64,
        "strict_certificate_sha256": "b" * 64,
        "observed_data_integrity_verified": True,
        "demonstrated_missing_required_observations": 0,
        "unresolved_span_count": 8,
        "unresolved_absent_minutes": 1051,
    }
    accepted = {
        "verified": True, "missing_expected_minutes": 0, "unknown_coverage": 0,
        "certificate_readiness": "NOT_FULLY_CERTIFIED",
        "calendar_coverage_certificate": {"readiness": "NOT_FULLY_CERTIFIED",
                                          "calendar_snapshots": [{"identity": "test"}]},
        "paper_coverage_acceptance": paper_policy,
    }
    runner = CausalBootstrapRunner(
        context=CausalMarketContext(), checkpoint_path=bootstrap_test_root / "accepted.json",
        activation_timestamp=activation, history_origin_timestamp=raw.timestamp.iloc[0],
        feature_cache_key="accepted-paper-fixture", initial_equity=INITIAL_EQUITY,
        risk_configuration=asdict(XFA_50K_PRODUCTION_POLICY),
        coverage_validator=lambda _: accepted,
    )
    _, identity, returned_coverage = runner._validate_inputs(raw, features)
    assert returned_coverage["certificate_readiness"] == "NOT_FULLY_CERTIFIED"
    assert returned_coverage["paper_coverage_acceptance"] == paper_policy
    assert identity["source"]["coverage_certificate_sha256"]

    rejected = CausalBootstrapRunner(
        context=CausalMarketContext(), checkpoint_path=bootstrap_test_root / "wrong-scope.json",
        activation_timestamp=activation, history_origin_timestamp=raw.timestamp.iloc[0],
        feature_cache_key="wrong-scope-fixture", initial_equity=INITIAL_EQUITY,
        risk_configuration=asdict(XFA_50K_PRODUCTION_POLICY),
        coverage_validator=lambda _: {**accepted, "paper_coverage_acceptance": {
            **paper_policy, "scope": "live_execution"}},
    )
    with pytest.raises(ValueError, match="coverage is not fully verified"):
        rejected._validate_inputs(raw, features)


def test_bootstrap_checkpoint_rejects_corruption_and_incompatible_identity(warmed_artifact):
    store = CausalBootstrapCheckpointStore(warmed_artifact["path"])
    with pytest.raises(ValueError, match="identity differs"):
        store.load(expected_identity={"different": True})

    original = warmed_artifact["path"].read_text(encoding="utf-8")
    try:
        warmed_artifact["path"].write_text(original[:-2] + "xx", encoding="utf-8")
        with pytest.raises((ValueError, json.JSONDecodeError)):
            store.load(expected_identity=warmed_artifact["identity"])
    finally:
        warmed_artifact["path"].write_text(original, encoding="utf-8")


def test_bootstrap_rejects_multithreaded_numerical_runtime(monkeypatch):
    import threadpoolctl

    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{
        "user_api": "blas", "internal_api": "openblas", "num_threads": 8,
        "version": "test",
    }])
    with pytest.raises(RuntimeError, match="single-thread numerical pools"):
        _numerical_thread_identity()


def test_fresh_strategy_seed_is_constructor_default_not_historical_execution():
    from src.paper.causal_bootstrap import _fresh_strategy_state_seed

    seed = _fresh_strategy_state_seed()
    assert seed["initialization"] == "fresh_strategy_constructors_at_activation"
    assert seed["no_historical_strategy_execution"] is True
    assert set(seed["strategies"]) == {"MRL1", "MRS2", "S2R", "ORB"}
    assert seed["strategies"]["S2R"]["state"]["recovery_tracker"]["state"] == "initial"
    assert seed["strategies"]["ORB"]["state"]["orb_context"] is None


def test_activation_artifact_schema_rejects_unfitted_or_inherited_state(warmed_artifact):
    payload = json.loads(json.dumps(warmed_artifact["payload"], default=str))
    payload["account_seed"]["positions"] = [{"symbol": "MNQ"}]
    payload["context"]["hmm_provider"]["s2r"]["fit_count"] = 0
    errors = activation_payload_errors(payload, activation_timestamp=warmed_artifact["activation"])
    assert any("S2R causal HMM has no fitted model" in error for error in errors)
    assert any("must not inherit positions" in error for error in errors)


def test_warm_context_seeds_a_fresh_paper_account_without_inheriting_trades(warmed_artifact, bootstrap_test_root):
    from src.paper.run_autonomous import build_real_paper_engine

    engine, adapter = build_real_paper_engine(bootstrap_test_root / "fresh_account", initial_equity=INITIAL_EQUITY)
    try:
        seed = load_context_seed_for_new_account(
            store=CausalBootstrapCheckpointStore(warmed_artifact["path"]),
            context=adapter.context,
            expected_identity=warmed_artifact["identity"],
            activation_timestamp=warmed_artifact["activation"],
        )
        assert seed["initial_equity"] == INITIAL_EQUITY
        assert seed["risk_configuration"] == asdict(XFA_50K_PRODUCTION_POLICY)
        assert adapter.context.last_timestamp < warmed_artifact["activation"]
        assert engine.account_equity == INITIAL_EQUITY
        assert engine.execution.get_positions() == []
        assert engine.risk._open_positions == {}
        assert engine.broker._orders == {}
        assert engine.broker._positions == {}
        assert all(strategy._trade_state is None for strategy in engine.strategies
                   if hasattr(strategy, "_trade_state"))
    finally:
        pass
