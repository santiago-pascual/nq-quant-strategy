"""Explicit, hash-bound acceptance of residual history risk for simulated Paper.

This policy never changes the strict calendar certificate. It is limited to
internal simulated Paper and accepts only the unresolved spans explicitly
listed by the pinned certificate; any source or inventory change invalidates
the record.
"""
from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

POLICY_NAME = "paper_research_quality_accepted"
SCHEMA_VERSION = 1
EXPECTED_UNRESOLVED_SPANS = 8
EXPECTED_UNCERTAIN_MINUTES = 1051
ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(document: Mapping[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_condition_metadata(certificate: Mapping[str, Any]) -> dict[str, Any]:
    quality = certificate.get("source_quality_metadata", {})
    verified: dict[str, Any] = {}
    for label, path_key, sha_key, size_key in (
        ("condition", "path", "sha256", "bytes"),
        ("manifest", "manifest_path", "manifest_sha256", None),
    ):
        relative = quality.get(path_key)
        expected_hash = quality.get(sha_key)
        if not relative or not expected_hash:
            raise ValueError(f"coverage certificate lacks verified {label} metadata")
        path = (ROOT / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Databento {label} metadata does not match the strict certificate")
        if size_key and path.stat().st_size != int(quality.get(size_key, -1)):
            raise ValueError(f"Databento {label} metadata size differs from the strict certificate")
        verified[label] = {"path": str(path.relative_to(ROOT)), "sha256": expected_hash,
                           "bytes": path.stat().st_size}
    return verified


def _unresolved_spans(certificate_path: Path, certificate: Mapping[str, Any]) -> list[dict[str, Any]]:
    gap_path = certificate_path.with_name(certificate_path.stem + "_gaps.csv")
    if not gap_path.is_file():
        raise ValueError(f"strict gap inventory is missing: {gap_path}")
    gaps = pd.read_csv(gap_path)
    selected = gaps.loc[gaps["classification"] == "unresolved_source_quality_or_calendar"]
    spans: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        counts = ast.literal_eval(str(row["minute_class_counts"]))
        if not isinstance(counts, dict) or sum(int(value) for value in counts.values()) != int(row["missing_minutes"]):
            raise ValueError("unresolved gap inventory has inconsistent minute counts")
        spans.append({
            "previous_observed_utc": str(row["previous_observed_utc"]),
            "next_observed_utc": str(row["next_observed_utc"]),
            "elapsed_minutes": int(row["elapsed_minutes"]),
            "missing_minutes": int(row["missing_minutes"]),
            "classification": str(row["classification"]),
            "minute_class_counts": {str(k): int(v) for k, v in counts.items()},
        })
    expected_count = sum(item["minute_class_counts"].get("degraded_source_day_gap_unresolved", 0)
                         for item in spans)
    if len(spans) != EXPECTED_UNRESOLVED_SPANS or expected_count != EXPECTED_UNCERTAIN_MINUTES:
        raise ValueError(
            f"unresolved scope changed: found {len(spans)} spans/{expected_count} "
            "uncertain minutes (the rows also include already-classified closures); "
            f"expected {EXPECTED_UNRESOLVED_SPANS}/{EXPECTED_UNCERTAIN_MINUTES}"
        )
    recorded_counts = certificate.get("missing_minute_classification_counts", {})
    if int(recorded_counts.get("degraded_source_day_gap_unresolved", -1)) != expected_count:
        raise ValueError("strict certificate unresolved-minute summary differs from its gap inventory")
    return spans


def _paper_acceptance_decision(validation: Mapping[str, Any],
                               certificate: Mapping[str, Any],
                               spans: list[dict[str, Any]]) -> dict[str, Any]:
    if validation.get("status") != "VALID" or validation.get("observed_data_integrity") is not True:
        raise ValueError("strict observed-data/source integrity validation failed")
    coverage = validation.get("coverage", {})
    if int(coverage.get("demonstrated_missing_required_observations", -1)) != 0:
        raise ValueError("demonstrated missing required observations cannot be risk-accepted")
    uncertain_minutes = sum(
        x["minute_class_counts"].get("degraded_source_day_gap_unresolved", 0) for x in spans
    )
    if len(spans) != EXPECTED_UNRESOLVED_SPANS or uncertain_minutes != EXPECTED_UNCERTAIN_MINUTES:
        raise ValueError("accepted residual scope differs from the reviewed eight-span inventory")
    if certificate.get("readiness") == "FULLY_CERTIFIED":
        raise ValueError("acceptance record is only for the known residual-risk certificate")
    return {
        "policy": POLICY_NAME,
        "decision": "ACCEPTED_FOR_INTERNAL_SIMULATED_PAPER_ONLY",
        "strict_certificate_readiness": certificate.get("readiness"),
        "activation_coverage_verified": False,
        "observed_data_integrity_verified": True,
        "demonstrated_missing_required_observations": 0,
        "unresolved_span_count": len(spans),
        "unresolved_absent_minutes": uncertain_minutes,
    }


def build_acceptance_record(certificate_path: str | Path) -> dict[str, Any]:
    """Build the exact acceptance payload after rerunning strict validation."""
    from src.paper.causal_bootstrap_cli import validate_inputs

    path = Path(certificate_path).resolve()
    certificate = json.loads(path.read_text(encoding="utf-8"))
    strict_validation = validate_inputs(path, check_files=True)
    spans = _unresolved_spans(path, certificate)
    decision = _paper_acceptance_decision(strict_validation, certificate, spans)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY_NAME,
        "scope": "internal_simulated_paper_only_no_live_execution",
        "accepted_by": "explicit_user_authorization_in_active_task",
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_is_cryptographic_signature": False,
        "statement": (
            "The user accepts the documented residual Databento degraded-source/calendar "
            "uncertainty for simulated Paper bootstrap only. The strict certificate remains "
            "NOT_FULLY_CERTIFIED. No missing bars or trades are inferred or synthesized."
        ),
        "strict_certificate": {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "readiness": certificate.get("readiness"),
            "gap_inventory_path": str(path.with_name(path.stem + "_gaps.csv").relative_to(ROOT)),
            "gap_inventory_sha256": sha256_file(path.with_name(path.stem + "_gaps.csv")),
        },
        "source_condition_metadata": _verify_condition_metadata(certificate),
        "source_data_file_manifest": [
            {"path": item["path"], "bytes": int(item["bytes"]), "sha256": item["sha256"]}
            for item in certificate.get("data_files", [])
        ],
        "accepted_degraded_dates": list(certificate.get("source_quality_metadata", {}).get(
            "degraded_dates_with_observed_mnq_rows", [])),
        "accepted_unresolved_spans": spans,
        "strict_validation_summary": strict_validation,
        "decision": decision,
    }


def write_acceptance_record(certificate_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    record = build_acceptance_record(certificate_path)
    record["acceptance_payload_sha256"] = _canonical_hash(record)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output.parent, prefix=output.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return record


def validate_acceptance_record(acceptance_path: str | Path,
                               certificate_path: str | Path) -> dict[str, Any]:
    acceptance_file = Path(acceptance_path).resolve()
    certificate_file = Path(certificate_path).resolve()
    stored = json.loads(acceptance_file.read_text(encoding="utf-8"))
    expected = build_acceptance_record(certificate_file)
    digest = stored.get("acceptance_payload_sha256")
    payload = {key: value for key, value in stored.items() if key != "acceptance_payload_sha256"}
    if digest != _canonical_hash(payload):
        raise ValueError("Paper coverage acceptance artifact checksum mismatch")
    immutable_keys = (
        "schema_version", "policy", "scope", "accepted_by", "acceptance_is_cryptographic_signature",
        "strict_certificate", "source_condition_metadata", "source_data_file_manifest",
        "accepted_degraded_dates", "accepted_unresolved_spans", "decision",
    )
    for key in immutable_keys:
        if stored.get(key) != expected.get(key):
            raise ValueError(f"Paper coverage acceptance no longer matches current strict evidence: {key}")
    if stored.get("policy") != POLICY_NAME or stored.get("scope") != "internal_simulated_paper_only_no_live_execution":
        raise ValueError("Paper coverage acceptance has an invalid policy or scope")
    return {"valid": True, "policy": POLICY_NAME,
            "decision": stored["decision"],
            "acceptance_artifact_sha256": sha256_file(acceptance_file),
            "strict_certificate_readiness": stored["strict_certificate"]["readiness"],
            "strict_certificate_sha256": stored["strict_certificate"]["sha256"],
            "strict_certificate_path": stored["strict_certificate"]["path"],
            "source_files_match": True,
            "accepted_unresolved_spans": stored["accepted_unresolved_spans"],
            "accepted_degraded_dates": stored["accepted_degraded_dates"],
            "scope": stored["scope"],
            "observed_data_integrity_verified": stored["decision"]["observed_data_integrity_verified"],
            "demonstrated_missing_required_observations": stored["decision"]["demonstrated_missing_required_observations"],
            "unresolved_span_count": stored["decision"]["unresolved_span_count"],
            "unresolved_absent_minutes": stored["decision"]["unresolved_absent_minutes"]}
