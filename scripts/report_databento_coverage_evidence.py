"""Summarize local Databento batch provenance alongside unresolved MNQ gaps.

This report verifies the downloaded package/query metadata and day-level
Databento condition status. It deliberately does not promote a date or absent
OHLCV minute to fully covered: daily availability is not a per-symbol,
per-minute completeness guarantee for trade-derived bars.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raw/mnq/ohlcv_1m"
DEFAULT_CERT = ROOT / "results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_v2_recurring_rules.json"
DEFAULT_GAPS = ROOT / "results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_v2_recurring_rules_gaps.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _days_between(first: date, last: date):
    current = first
    while current <= last:
        yield current
        current += timedelta(days=1)


def build_evidence_report(data_dir: Path, certificate_path: Path, gaps_path: Path) -> dict[str, Any]:
    import pandas as pd

    metadata_path, manifest_path, condition_path = (
        data_dir / "metadata.json", data_dir / "manifest.json", data_dir / "condition.json"
    )
    for path in (metadata_path, manifest_path, condition_path, certificate_path, gaps_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    query = metadata.get("query", {})
    if (query.get("dataset") != "GLBX.MDP3" or query.get("schema") != "ohlcv-1m"
            or query.get("symbols") != ["MNQ.v.0"]
            or query.get("stype_in") != "continuous"
            or query.get("stype_out") != "instrument_id"):
        raise ValueError("Local Databento query identity does not match expected MNQ.v.0 request")
    if manifest.get("job_id") != metadata.get("job_id"):
        raise ValueError("Batch manifest and query metadata job IDs differ")

    file_checks = []
    missing_files = []
    hash_mismatches = []
    for entry in manifest.get("files", []):
        filename = entry.get("filename")
        if not filename:
            continue
        path = data_dir / filename
        if not path.is_file():
            missing_files.append(filename)
            continue
        actual = _sha256(path)
        expected = str(entry.get("hash", "")).removeprefix("sha256:")
        row = {"filename": filename, "bytes": path.stat().st_size,
               "expected_sha256": expected, "actual_sha256": actual,
               "hash_matches": actual == expected}
        file_checks.append(row)
        if actual != expected:
            hash_mismatches.append(filename)

    condition_sources = [condition_path]
    staging_dir = data_dir.parent / "staging"
    package_dirs = [data_dir]
    if staging_dir.is_dir():
        for stage in sorted(staging_dir.iterdir()):
            extracted = stage / "extracted"
            if extracted.is_dir() and (extracted / "condition.json").is_file():
                condition_sources.append(extracted / "condition.json")
                package_dirs.append(extracted)
    condition_by_day = {}
    condition_conflicts = []
    for source in condition_sources:
        conditions = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(conditions, list):
            raise ValueError(f"Databento condition metadata must be a list: {source}")
        for row in conditions:
            day = pd.Timestamp(row["date"]).date()
            prior = condition_by_day.get(day)
            if prior is not None and prior.get("condition") != row.get("condition"):
                condition_conflicts.append({"date": day.isoformat(),
                                            "prior": prior.get("condition"),
                                            "new": row.get("condition"),
                                            "source": str(source)})
            condition_by_day[day] = row
    input_range = certificate.get("input", {})
    first_text = certificate.get("data_first_bar_utc", input_range.get("first_utc"))
    last_text = certificate.get("data_last_bar_utc", input_range.get("last_utc"))
    if not first_text or not last_text:
        raise ValueError("Coverage certificate has no observed first/last timestamp")
    first = pd.Timestamp(first_text).date()
    last = pd.Timestamp(last_text).date()
    covered_conditions = [condition_by_day[d] for d in _days_between(first, last)
                          if d in condition_by_day]
    condition_counts = Counter(str(row.get("condition", "unknown")) for row in covered_conditions)
    degraded = {pd.Timestamp(row["date"]).date() for row in covered_conditions
                if row.get("condition") == "degraded"}

    gaps = pd.read_csv(gaps_path)
    uncertain_dates: set[date] = set()
    degraded_minutes_by_date: Counter[str] = Counter()
    span_intervening_minutes_by_class: Counter[str] = Counter()
    gap_spans_by_class: Counter[str] = Counter()
    for row in gaps.itertuples(index=False):
        classification = str(row.classification)
        gap_spans_by_class[classification] += 1
        span_intervening_minutes_by_class[classification] += int(row.missing_minutes)
    degraded_uncertain_minutes = 0
    degraded_gap_spans = 0
    for row in gaps.itertuples(index=False):
        if not str(row.classification).startswith("uncertain"):
            continue
        prev = pd.Timestamp(row.previous_observed_utc)
        nxt = pd.Timestamp(row.next_observed_utc)
        if prev.tzinfo is None or nxt.tzinfo is None:
            raise ValueError("Gap timestamps must include timezone offsets")
        minute = prev.floor("min") + pd.Timedelta(minutes=1)
        stop = nxt.floor("min")
        span_degraded = False
        while minute < stop:
            if minute.date() in degraded:
                degraded_uncertain_minutes += 1
                uncertain_dates.add(minute.date())
                degraded_minutes_by_date[minute.date().isoformat()] += 1
                span_degraded = True
            minute += pd.Timedelta(minutes=1)
        degraded_gap_spans += int(span_degraded)

    package_checks = []
    for package in package_dirs:
        package_metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
        package_manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        package_query = package_metadata.get("query", {})
        if (package_query.get("dataset") != "GLBX.MDP3"
                or package_query.get("schema") != "ohlcv-1m"
                or package_query.get("symbols") != ["MNQ.v.0"]
                or package_query.get("stype_in") != "continuous"
                or package_manifest.get("job_id") != package_metadata.get("job_id")):
            raise ValueError(f"Staged Databento package identity invalid: {package}")
        listed = []
        for entry in package_manifest.get("files", []):
            filename = entry.get("filename")
            if not filename:
                continue
            local = package / filename
            if not local.is_file():
                listed.append({"filename": filename, "present": False, "hash_matches": False})
                continue
            expected = str(entry.get("hash", "")).removeprefix("sha256:")
            actual = _sha256(local)
            listed.append({"filename": filename, "present": True,
                           "hash_matches": actual == expected})
        package_checks.append({"path": str(package), "job_id": package_metadata.get("job_id"),
                               "query": package_query,
                               "all_manifest_files_present_and_matching": bool(listed) and all(
                                   row["present"] and row["hash_matches"] for row in listed),
                               "manifest_file_checks": listed})

    partition_checks = []
    for entry in certificate.get("data_files", []):
        relative = str(entry.get("path", "")).replace("\\", "/")
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        expected = str(entry.get("sha256", ""))
        partition_checks.append({"path": relative, "present": path.is_file(),
                                 "hash_matches": actual == expected if actual else False})
    import_manifest_checks = []
    if staging_dir.is_dir():
        for stage in sorted(staging_dir.iterdir()):
            for name in ("import_manifest.json", "incremental_import_manifest.json"):
                path = stage / name
                if not path.is_file():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                relative_target = record.get("target_relative_path")
                output_name = record.get("output_filename")
                target = str(relative_target or output_name).replace("\\", "/") if (relative_target or output_name) else None
                if relative_target:
                    target_path = ROOT / target
                elif output_name:
                    target_path = data_dir / output_name
                else:
                    target_path = None
                expected_output = record.get("output_sha256")
                actual_output = _sha256(target_path) if target_path and target_path.is_file() else None
                source_name = record.get("source_batch_file") or record.get("source_filename")
                source_candidates = [path.parent / str(source_name),
                                     path.parent / "extracted" / str(source_name)] if source_name else []
                source_path = next((candidate for candidate in source_candidates if candidate.is_file()), None)
                expected_source = record.get("source_batch_sha256") or record.get("source_sha256")
                actual_source = _sha256(source_path) if source_path else None
                import_manifest_checks.append({
                    "manifest": str(path.relative_to(ROOT)),
                    "job_id": record.get("job_id"), "target": target,
                    "target_present": bool(target_path and target_path.is_file()),
                    "expected_output_sha256": expected_output,
                    "actual_output_sha256": actual_output,
                    "output_hash_matches": (actual_output.lower() == str(expected_output).lower()
                                            if expected_output and actual_output else None),
                    "source_file": str(source_path.relative_to(ROOT)) if source_path else None,
                    "expected_source_sha256": expected_source,
                    "actual_source_sha256": actual_source,
                    "source_hash_matches": (actual_source.lower() == str(expected_source).removeprefix("sha256:").lower()
                                            if expected_source and actual_source else None),
                    "source_identity": {k: record.get(k) for k in
                                        ("dataset", "schema", "symbols", "mapping", "source_batch_sha256")
                                        if k in record},
                })

    expected_query_files = [entry for entry in manifest.get("files", [])
                            if str(entry.get("filename", "")).endswith(".csv.zst")]
    return {
        "report_schema": 1,
        "source": "local Databento batch artifacts; no network request made",
        "job_id": metadata.get("job_id"),
        "dataset": query.get("dataset"), "schema": query.get("schema"),
        "symbols": query.get("symbols"), "stype_in": query.get("stype_in"),
        "stype_out": query.get("stype_out"),
        "query_start_ns": query.get("start"), "query_end_ns_exclusive": query.get("end"),
        "customizations": metadata.get("customizations", {}),
        "coverage_interval_utc": {"start": first_text, "end_inclusive": last_text},
        "batch_integrity": {
            "listed_data_files": len(expected_query_files), "checked_files": len(file_checks),
            "missing_files": missing_files, "hash_mismatches": hash_mismatches,
            "all_manifest_files_present_and_matching": not missing_files and not hash_mismatches,
            "checks": file_checks,
        },
        "vendor_day_condition_metadata": {
            "date_count_in_bar_coverage": len(covered_conditions),
            "status_counts": dict(sorted(condition_counts.items())),
            "degraded_dates": sorted(day.isoformat() for day in degraded),
            "conflicts": condition_conflicts,
            "interpretation": (
                "Databento daily condition metadata records dataset-level date condition. "
                "It does not establish completeness for every continuous-symbol OHLCV minute, "
                "nor distinguish no-trade minutes from absent capture."
            ),
        },
        "uncertain_gap_overlap_with_degraded_dates": {
            "gap_span_count": degraded_gap_spans,
            "absent_minute_count": degraded_uncertain_minutes,
            "dates": sorted(day.isoformat() for day in uncertain_dates),
            "absent_minutes_by_utc_date": dict(sorted(degraded_minutes_by_date.items())),
            "classification_effect": "diagnostic_only; gaps remain uncertain and bootstrap activation remains blocked",
        },
        "historical_absence_evidence": {
            "gap_spans_by_classification": dict(sorted(gap_spans_by_class.items())),
            "intervening_wall_clock_minutes_inside_gap_spans_by_classification": dict(
                sorted(span_intervening_minutes_by_class.items())
            ),
            "coverage_certificate_missing_minute_classes": certificate.get(
                "missing_minute_classification_counts", {}
            ),
            "uncertain_open_or_exception_absent_minutes": int(
                certificate.get("missing_minute_classification_counts", {}).get(
                    "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown", 0
                )
            ),
            "demonstrated_missing_required_observations": 0,
            "required_observation_interpretation": (
                "No exact missing trade/record has been demonstrated by available local evidence. "
                "This is not evidence that none are missing; OHLCV trade bars omit no-trade minutes "
                "and date-level condition metadata cannot resolve capture completeness."
            ),
            "legitimate_sparse_trade_ohlcv": {
                "distinguishable_from_capture_loss": False,
                "classification": "UNRESOLVED_WITHOUT_TRADE_OR_SEQUENCE_LEVEL_EVIDENCE",
            },
            "degraded_source_dates": sorted(day.isoformat() for day in degraded),
            "degraded_dates_with_uncertain_absences": sorted(day.isoformat() for day in uncertain_dates),
            "degraded_uncertain_absent_minute_count": degraded_uncertain_minutes,
            "degraded_condition_is_not_a_completeness_finding": True,
        },
        "integrated_source_lineage": {
            "batch_packages": package_checks,
            "certificate_partition_hashes": partition_checks,
            "import_manifests": import_manifest_checks,
            "limitation": (
                "An import manifest without a final-output hash proves recorded procedure and mapping, "
                "not byte-for-byte identity of the canonicalized output partition."
            ),
        },
        "coverage_conclusion": {
            "request_identity_and_local_batch_integrity_verified": (
                not missing_files and not hash_mismatches
                and all(item["all_manifest_files_present_and_matching"] for item in package_checks)
                and bool(partition_checks)
                and all(item["present"] and item["hash_matches"] for item in partition_checks)
            ),
            "per_minute_source_completeness_proven": False,
            "readiness": "NOT_FULLY_CERTIFIED",
            "reason": (
                "Manifest/query metadata and file hashes verify which request was downloaded and "
                "that local package files match the manifest. Daily available/degraded status is "
                "not a per-instrument record completeness attestation; trade-derived OHLCV cannot "
                "separate no-trade minutes from capture loss."
            ),
            "additional_evidence_needed": [
                "Databento per-symbol/per-interval completeness or sequence-gap attestation for the exact batch, "
                "or independent trade/event data sufficient to distinguish no trades from missing capture.",
                "Date-specific authoritative CME schedules for remaining holiday/early-close intervals, "
                "if those intervals are to be certified as scheduled closures rather than left uncertain.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERT)
    parser.add_argument("--gaps", type=Path, default=DEFAULT_GAPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_evidence_report(args.data_dir, args.certificate, args.gaps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "batch_integrity": report["batch_integrity"]["all_manifest_files_present_and_matching"],
                      "day_condition_status_counts": report["vendor_day_condition_metadata"]["status_counts"],
                      "degraded_uncertain_absent_minutes": report["uncertain_gap_overlap_with_degraded_dates"]["absent_minute_count"],
                      "readiness": report["coverage_conclusion"]["readiness"]}, indent=2))
    return 0 if report["batch_integrity"]["all_manifest_files_present_and_matching"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
