"""Versioned runtime identity validation for Paper checkpoint restore.

Version 2 separates execution-critical source/dependency identity from the
read-only analytics/reporting module. Legacy v1 identities are accepted only
through the single exact migration profile below; future analytics changes are
not implicitly compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = 2
REPORTING_SCHEMA_VERSION = 1
COMPATIBILITY_POLICY_VERSION = 1
ANALYTICS_MIGRATION_ID = "paper-analytics-reporting-isolation-v1"

LEGACY_V1_ANALYTICS_SHA256 = "354a2b8a45dcd87a262bf9d44ebe211cecec0d7accf801913651c8016457d2db"
CURRENT_V2_ANALYTICS_SHA256 = "26132abb1917152e6195568bcdef1a9eccc5d5280ba2b1f59d0122a9ee4fe78e"
APPROVED_POLICY_CODE_SHA256 = "92bc1abf3c5a1c50288c7d06e9e2bfdb3ca00c31ea258710f01e2dcc7704fae4"

# These two files change only to implement and invoke the versioned identity
# migration. Every other legacy execution source remains byte-for-byte strict.
# Values are filled with the reviewed post-change digests before validation.
LEGACY_V1_MIGRATED_SOURCE_SHA256 = {
    "src/paper/realtime_checkpoint.py": "ba55ef8edf2301e68ac6348d16cb6758c131025db728530e349609c925eca9e8",
    "src/paper/realtime_service.py": "f373fa7872755792c6fa0c3aec11a0033b9bec028902299c6b3cb2ed4fa1ef35",
    "src/paper/delayed_paper_cli.py": "0aa3b983e5dace29a6a1b5a8dd9b640749e098705e3c0831d44c84f26a60fc45",
}
CURRENT_V2_MIGRATED_SOURCE_SHA256 = {
    "src/paper/realtime_checkpoint.py": "3572c04d0e6893158673da31f538ff7291748b8f7a9a0ad469318a4a5f0fafc2",
    "src/paper/realtime_service.py": "ef99db31bc7b47b1ba318ed35772933b4ea5bf17ac5339eb0c276ee85f7e5ab7",
    "src/paper/delayed_paper_cli.py": "60f6d6ee1cbc396129fdf3d3e0b1f0b34c4437d8c20c2a6c400ec2f5e64faf11",
}
LEGACY_V1_NEW_SOURCE_SHA256 = {
    "src/paper/ibkr_paper_recovery.py": "22f875cde6ffd45baa6eac3eb132315457c3173e0f2c8209f8c2f1dfc9b6b3f0",
    "src/paper/runtime_compatibility.py": "92bc1abf3c5a1c50288c7d06e9e2bfdb3ca00c31ea258710f01e2dcc7704fae4",
}


@dataclass(frozen=True)
class RuntimeCompatibilityResult:
    accepted: bool
    mode: str
    policy_id: str | None
    reason: str
    execution_match: bool
    reporting_match: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "mode": self.mode,
            "policy_id": self.policy_id,
            "reason": self.reason,
            "execution_match": self.execution_match,
            "reporting_match": self.reporting_match,
        }


def _reject(reason: str) -> RuntimeCompatibilityResult:
    return RuntimeCompatibilityResult(False, "REJECTED", None, reason, False, False)


def compatibility_policy_sha256() -> str:
    """Hash policy implementation while normalizing its pinned digest values.

    This permits the policy module to be included in runtime identity without
    a self-referential hash. Any executable policy edit changes this digest.
    """
    path = Path(__file__)
    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r'(?m)^APPROVED_POLICY_CODE_SHA256 = "[^"]*"$',
        'APPROVED_POLICY_CODE_SHA256 = "<normalized>"', source,
    )
    source = re.sub(
        r'(?m)^CURRENT_V2_MIGRATED_SOURCE_SHA256 = \{.*?^\}$',
        'CURRENT_V2_MIGRATED_SOURCE_SHA256 = {<normalized>}', source, flags=re.S,
    )
    source = re.sub(
        r'(?m)^LEGACY_V1_NEW_SOURCE_SHA256 = \{.*?^\}$',
        'LEGACY_V1_NEW_SOURCE_SHA256 = {<normalized>}', source, flags=re.S,
    )
    return sha256(source.encode("utf-8")).hexdigest()


def validate_runtime_identity(saved: Any, current: Any) -> RuntimeCompatibilityResult:
    """Accept exact v2 identities or the one audited v1-to-v2 migration.

    The legacy path requires exact Python/dependency versions and exact hashes
    for every execution source except the explicitly mapped identity-migration
    files. The reporting file must match the exact saved/current analytics
    pair. Unknown keys, hashes, schemas, or future analytics versions fail
    closed.
    """
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        return _reject("runtime identity is not a mapping")
    if dict(saved) == dict(current):
        return RuntimeCompatibilityResult(True, "EXACT_V2", None, "runtime identities match exactly", True, True)
    if int(current.get("identity_schema_version", 0)) != IDENTITY_SCHEMA_VERSION:
        return _reject("current runtime identity schema is unsupported")
    if "identity_schema_version" in saved:
        return _reject("versioned runtime identity differs; exact v2 identity is required")

    saved_sources = saved.get("source_sha256")
    current_sources = current.get("source_sha256")
    execution = current.get("execution_fingerprint")
    reporting = current.get("reporting_fingerprint")
    if not all(isinstance(value, Mapping) for value in (saved_sources, current_sources, execution, reporting)):
        return _reject("legacy/current identity fields are incomplete")
    if saved.get("python") != current.get("python") or saved.get("packages") != current.get("packages"):
        return _reject("Python or numerical dependency identity differs")
    if (int(current.get("compatibility_policy_version", 0)) != COMPATIBILITY_POLICY_VERSION
            or int(reporting.get("schema_version", 0)) != REPORTING_SCHEMA_VERSION):
        return _reject("runtime compatibility/reporting schema is not the audited version")
    if compatibility_policy_sha256() != APPROVED_POLICY_CODE_SHA256:
        return _reject("runtime compatibility policy source is not the audited implementation")

    reporting_sources = reporting.get("source_sha256")
    execution_sources = execution.get("source_sha256")
    if not isinstance(reporting_sources, Mapping) or not isinstance(execution_sources, Mapping):
        return _reject("split execution/reporting fingerprints are incomplete")
    if saved_sources.get("src/paper/analytics.py") != LEGACY_V1_ANALYTICS_SHA256:
        return _reject("legacy analytics hash is not the explicitly audited checkpoint-era digest")
    if reporting_sources.get("src/paper/analytics.py") != CURRENT_V2_ANALYTICS_SHA256:
        return _reject("current analytics hash is not the explicitly audited reporting implementation")

    for path, legacy_hash in LEGACY_V1_MIGRATED_SOURCE_SHA256.items():
        if saved_sources.get(path) != legacy_hash:
            return _reject(f"legacy migration source does not match the approved predecessor: {path}")
        if current_sources.get(path) != CURRENT_V2_MIGRATED_SOURCE_SHA256.get(path):
            return _reject(f"current migration source differs from the audited implementation: {path}")
        if not CURRENT_V2_MIGRATED_SOURCE_SHA256.get(path):
            return _reject(f"migration profile is not finalized for {path}")

    saved_paths = set(saved_sources)
    expected_saved_paths = set(current_sources) - set(LEGACY_V1_NEW_SOURCE_SHA256)
    if saved_paths != expected_saved_paths:
        return _reject("legacy source manifest has missing or unexpected source paths")

    ignored_legacy_paths = {"src/paper/analytics.py", *LEGACY_V1_MIGRATED_SOURCE_SHA256}
    for path in sorted(saved_paths - ignored_legacy_paths):
        if saved_sources.get(path) != execution_sources.get(path):
            return _reject(f"execution-critical source fingerprint differs: {path}")

    for path, expected_hash in LEGACY_V1_NEW_SOURCE_SHA256.items():
        if not expected_hash or current_sources.get(path) != expected_hash:
            return _reject(f"new recovery source is not the audited implementation: {path}")

    return RuntimeCompatibilityResult(
        True, "LEGACY_V1_ANALYTICS_MIGRATION", ANALYTICS_MIGRATION_ID,
        "all execution fingerprints match; exact reporting-only migration accepted",
        True, False,
    )


def bootstrap_lineage_matches(bootstrap: Any, checkpoint: Any) -> bool:
    """Compare immutable bootstrap lineage with a v1 or v2 Paper checkpoint.

    The historical orchestrator and reporting hashes are excluded by their
    established non-execution classification. All other overlapping source
    hashes and numerical dependencies must match, including the explicit
    migration aliases when the checkpoint uses v2.
    """
    if not isinstance(bootstrap, Mapping) or not isinstance(checkpoint, Mapping):
        return False
    if bootstrap == checkpoint:
        return True
    old_sources = bootstrap.get("source_sha256")
    current_sources = checkpoint.get("source_sha256")
    if not isinstance(old_sources, Mapping) or not isinstance(current_sources, Mapping):
        return False
    if bootstrap.get("python") != checkpoint.get("python") or bootstrap.get("packages") != checkpoint.get("packages"):
        return False
    non_execution = {"src/paper/analytics.py", "src/paper/delayed_paper_cli.py"}
    migrations = set(LEGACY_V1_MIGRATED_SOURCE_SHA256)
    for path, old_hash in old_sources.items():
        if path in non_execution:
            continue
        if path in migrations:
            current_hash = current_sources.get(path)
            if current_hash == old_hash:
                continue
            if (old_hash == LEGACY_V1_MIGRATED_SOURCE_SHA256[path]
                    and current_hash == CURRENT_V2_MIGRATED_SOURCE_SHA256.get(path)):
                continue
            else:
                return False
            continue
        if current_sources.get(path) != old_hash:
            return False
    return True
