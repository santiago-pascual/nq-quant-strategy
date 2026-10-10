from __future__ import annotations

from copy import deepcopy

from src.paper.realtime_checkpoint import _encode, runtime_identity
from src.paper.runtime_compatibility import (
    LEGACY_V1_ANALYTICS_SHA256,
    LEGACY_V1_MIGRATED_SOURCE_SHA256,
    LEGACY_V1_NEW_SOURCE_SHA256,
    validate_runtime_identity,
)


def _legacy_identity(current: dict) -> dict:
    saved = {
        "python": current["python"],
        "packages": deepcopy(current["packages"]),
        "source_sha256": deepcopy(current["source_sha256"]),
    }
    for path in LEGACY_V1_NEW_SOURCE_SHA256:
        saved["source_sha256"].pop(path)
    saved["source_sha256"]["src/paper/analytics.py"] = LEGACY_V1_ANALYTICS_SHA256
    for path, digest in LEGACY_V1_MIGRATED_SOURCE_SHA256.items():
        saved["source_sha256"][path] = digest
    return saved


def test_exact_current_runtime_is_accepted() -> None:
    identity = runtime_identity()
    result = validate_runtime_identity(identity, identity)
    assert result.accepted
    assert result.mode == "EXACT_V2"
    assert result.execution_match and result.reporting_match


def test_only_exact_audited_analytics_migration_is_accepted() -> None:
    result = validate_runtime_identity(_legacy_identity(runtime_identity()), runtime_identity())
    assert result.accepted
    assert result.mode == "LEGACY_V1_ANALYTICS_MIGRATION"
    assert result.policy_id == "paper-analytics-reporting-isolation-v1"
    assert result.execution_match
    assert not result.reporting_match


def test_unapproved_analytics_hash_is_rejected() -> None:
    saved = _legacy_identity(runtime_identity())
    saved["source_sha256"]["src/paper/analytics.py"] = "0" * 64
    result = validate_runtime_identity(saved, runtime_identity())
    assert not result.accepted
    assert "analytics hash" in result.reason


def test_execution_critical_source_change_is_rejected() -> None:
    saved = _legacy_identity(runtime_identity())
    saved["source_sha256"]["src/models/causal_hmm.py"] = "0" * 64
    result = validate_runtime_identity(saved, runtime_identity())
    assert not result.accepted
    assert "execution-critical" in result.reason


def test_dependency_change_is_rejected() -> None:
    saved = _legacy_identity(runtime_identity())
    saved["packages"]["numpy"] = "0.0.0"
    result = validate_runtime_identity(saved, runtime_identity())
    assert not result.accepted
    assert "dependency identity" in result.reason


def test_unknown_identity_schema_fails_closed() -> None:
    current = runtime_identity()
    current["identity_schema_version"] = 999
    result = validate_runtime_identity(_legacy_identity(runtime_identity()), current)
    assert not result.accepted


def test_set_valued_execution_state_encodes_deterministically() -> None:
    assert _encode({"fill-b", "fill-a"}) == _encode(set(["fill-a", "fill-b"]))
