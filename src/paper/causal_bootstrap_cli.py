"""Fail-closed CLI for validating and running the existing causal bootstrap.

This is an orchestration layer around CausalBootstrapRunner; it does not
implement a second feature or HMM engine. ``benchmark`` is explicitly
non-activation work. ``start``/``resume`` require a complete reviewed coverage
certificate and an explicit operator flag.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CERTIFICATE = ROOT / "results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_v2_recurring_rules.json"
DEFAULT_CACHE = ROOT / "results/paper/causal_bootstrap_feature_cache"
DEFAULT_PROGRESS = ROOT / "results/paper/causal_bootstrap_progress.json"
DEFAULT_CHECKPOINT = ROOT / "results/paper/causal_activation_seed.json"
HISTORY_ORIGIN = "2019-05-05T22:03:00Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_identity_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_raw() -> pd.DataFrame:
    from src.paper.autonomous_runner import load_canonical_raw_mnq
    return load_canonical_raw_mnq(include_contract_metadata=True)


def _source_file_check(certificate: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    expected = certificate.get("data_files")
    if not isinstance(expected, list) or not expected:
        return False, ["certificate has no data-file manifest"]
    expected_paths = set()
    for record in expected:
        path = (ROOT / record["path"]).resolve()
        expected_paths.add(path)
        if not path.is_file():
            mismatches.append(f"missing source file: {record['path']}")
        elif path.stat().st_size != int(record.get("bytes", -1)):
            mismatches.append(f"source size changed: {record['path']}")
        elif _sha256(path) != record.get("sha256"):
            mismatches.append(f"source checksum changed: {record['path']}")
    actual_paths = set((ROOT / "data/raw/mnq/ohlcv_1m").glob("*.csv.zst"))
    extras = sorted(str(p.relative_to(ROOT)) for p in actual_paths - expected_paths)
    missing_from_manifest = sorted(str(p.relative_to(ROOT)) for p in expected_paths - actual_paths)
    if extras:
        mismatches.append(f"uncovered raw partitions are present: {extras}")
    if missing_from_manifest:
        mismatches.append(f"manifest partitions are absent: {missing_from_manifest}")
    return not mismatches, mismatches


def _coverage_summary(certificate: dict[str, Any]) -> dict[str, Any]:
    gap_classes = certificate.get("gap_classification_counts", {})
    minute_classes = certificate.get("missing_minute_classification_counts", {})
    source_quality = certificate.get("source_quality_metadata", {})
    recurring = sum(int(v) for k, v in minute_classes.items() if k.startswith("scheduled_recurring_"))
    reviewed = sum(int(v) for k, v in minute_classes.items() if k.startswith("scheduled_holiday") or k == "scheduled_closure")
    uncertain = sum(int(v) for k, v in minute_classes.items()
                    if k.startswith("unverified") or "unresolved" in k)
    # The current source has no independent capture/no-trade event record, so
    # no uncertain span is promoted to a demonstrated missing observation.
    demonstrated = int(certificate.get("demonstrated_missing_required_observations", 0))
    sparse_trade_minutes = int(minute_classes.get(
        "trade_derived_ohlcv_minute_absence_not_required_by_grid_contract", 0
    ))
    degraded_source_minutes = int(minute_classes.get("degraded_source_day_gap_unresolved", 0))
    degraded_dates = source_quality.get("degraded_dates_in_observed_range", [])
    return {
        "verified_exchange_closure_gap_spans": sum(
            int(v) for k, v in gap_classes.items() if k == "verified_scheduled_closure"
        ),
        "uncertain_gap_spans": sum(
            int(v) for k, v in gap_classes.items() if "uncertain" in k or "unresolved" in k
        ),
        "recurring_closure_minutes": recurring,
        "reviewed_exception_closure_minutes": reviewed,
        "uncertain_absent_open_or_exception_minutes": uncertain,
        "sparse_trade_ohlcv_minutes_not_required_by_minute_grid": sparse_trade_minutes,
        "degraded_source_day_gap_minutes": degraded_source_minutes,
        "degraded_source_date_count": len(degraded_dates),
        "degraded_source_dates": degraded_dates,
        "demonstrated_missing_required_observations": demonstrated,
        "observed_rows": certificate.get("input", {}).get("rows"),
        "first_observed_utc": certificate.get("input", {}).get("first_utc"),
        "last_observed_utc": certificate.get("input", {}).get("last_utc"),
        "interpretation": (
            "OHLCV absence alone cannot distinguish no-trade minutes from capture loss. "
            "Uncertain spans remain explicit; they are not synthesized or certified."
        ),
    }


def validate_inputs(certificate_path: Path, *, check_files: bool = True) -> dict[str, Any]:
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if int(certificate.get("schema_version", 0)) < 2:
        mismatches.append("calendar/data certificate schema v2 or newer is required")
    if certificate.get("input", {}).get("first_utc") != HISTORY_ORIGIN.replace("Z", "+00:00"):
        # Accept equivalent offsets, then compare parsed UTC below.
        try:
            if pd.Timestamp(certificate["input"]["first_utc"]).tz_convert("UTC") != pd.Timestamp(HISTORY_ORIGIN):
                mismatches.append("certificate does not begin at the frozen historical origin")
        except Exception:
            mismatches.append("certificate first timestamp is invalid")
    elif pd.Timestamp(certificate["input"]["first_utc"]).tz_convert("UTC") != pd.Timestamp(HISTORY_ORIGIN):
        mismatches.append("certificate does not begin at the frozen historical origin")
    if certificate.get("input", {}).get("duplicate_timestamps") != 0:
        mismatches.append("certificate reports duplicate timestamps")
    if certificate.get("input", {}).get("nonmonotonic_transitions") != 0:
        mismatches.append("certificate reports nonmonotonic timestamps")
    source_files_match, file_mismatches = _source_file_check(certificate) if check_files else (None, [])
    mismatches.extend(file_mismatches)
    coverage = _coverage_summary(certificate)
    snapshots = certificate.get("calendar_snapshots", [])
    calendar_evidence_reviewed = bool(snapshots) and all(
        str(snapshot.get("review_status", "")).startswith("REVIEWED")
        and snapshot.get("identity")
        and snapshot.get("source_url")
        for snapshot in snapshots
    )
    if not calendar_evidence_reviewed:
        mismatches.append("calendar snapshot review provenance is incomplete")
    observed_integrity = not mismatches
    activation_coverage_verified = (
        certificate.get("readiness") == "FULLY_CERTIFIED"
        and calendar_evidence_reviewed
        and coverage["uncertain_gap_spans"] == 0
        and coverage["uncertain_absent_open_or_exception_minutes"] == 0
        and coverage["degraded_source_date_count"] == 0
        and coverage["demonstrated_missing_required_observations"] == 0
    )
    causal_computation_possible = (
        observed_integrity and coverage["demonstrated_missing_required_observations"] == 0
    )
    report_status = (
        "INVALID" if not observed_integrity
        else "OBSERVED_VALID_WITH_DEMONSTRATED_REQUIRED_GAPS"
        if coverage["demonstrated_missing_required_observations"] > 0
        else "VALID"
    )
    blockers = []
    if coverage["demonstrated_missing_required_observations"] > 0:
        blockers.append("certificate reports demonstrated missing required observations")
    if not activation_coverage_verified:
        if coverage["sparse_trade_ohlcv_minutes_not_required_by_minute_grid"]:
            blockers.append(
                f"{coverage['sparse_trade_ohlcv_minutes_not_required_by_minute_grid']} absent minutes are schema-sparse OHLCV, not required zero-trade rows; these are excluded from the minute-grid missing count"
            )
        if coverage["degraded_source_date_count"]:
            blockers.append(
                f"Databento marks {coverage['degraded_source_date_count']} observed-range date(s) degraded; its date-level condition is not MNQ completeness evidence, so those source intervals remain unresolved"
            )
        if coverage["uncertain_gap_spans"]:
            blockers.append(
                f"{coverage['uncertain_gap_spans']} gap spans remain unresolved ({coverage['uncertain_absent_open_or_exception_minutes']} absent open/exception minutes); independent source/session evidence is required"
            )
        if certificate.get("readiness") != "FULLY_CERTIFIED" and not blockers:
            blockers.append("historical calendar/source coverage certificate is not fully certified")
    return {
        "status": report_status,
        "certificate_path": str(certificate_path),
        "certificate_sha256": _sha256(certificate_path),
        "certificate_readiness": certificate.get("readiness"),
        "source_files_match_certificate": source_files_match,
        "observed_data_integrity": observed_integrity,
        "coverage": coverage,
        "causal_computation_possible_on_observed_bars": causal_computation_possible,
        "activation_coverage_verified": activation_coverage_verified,
        "activation_ready": observed_integrity and activation_coverage_verified,
        "mismatches": mismatches,
        "blockers": blockers,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()
                if not key.endswith("_seconds") and key not in {"written_at_utc", "fit_started_at_utc"}}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _windows_memory_counters() -> dict[str, int] | None:
    """Read process peak working set from Windows without adding a dependency."""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_memory_info = psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory_info.restype = wintypes.BOOL
        process = get_current_process()
        ok = get_memory_info(process, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return {"working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "peak_pagefile_bytes": int(counters.PeakPagefileUsage)}
    except Exception:
        return None


def _first_difference(left: Any, right: Any, path: str = "state") -> str | None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return f"{path}: key mismatch"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for i, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{i}]")
            if difference:
                return difference
        return None
    if isinstance(left, float) and isinstance(right, float) and pd.isna(left) and pd.isna(right):
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def run_benchmark(raw: pd.DataFrame, *, start: str, end: str, cache_dir: Path,
                  split_rows: int | None = None, progress_writer: Any | None = None) -> dict[str, Any]:
    from src.paper.bootstrap_artifacts import load_or_build_causal_feature_cache
    from src.paper.market_context import CausalMarketContext

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts.tzinfo is None or end_ts.tzinfo is None or start_ts >= end_ts:
        raise ValueError("benchmark requires an ordered, timezone-aware half-open interval")
    stamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    selected = raw.loc[(stamps >= start_ts.tz_convert("UTC")) & (stamps < end_ts.tz_convert("UTC"))].reset_index(drop=True)
    if selected.empty or len(selected) < 500:
        raise ValueError("benchmark interval must contain at least 500 observed bars")
    benchmark_started = time.perf_counter()
    cpu_started = time.process_time()
    timing: dict[str, float] = {}
    tick = time.perf_counter()
    if progress_writer is not None:
        progress_writer.stage("benchmark_feature_cache", total_rows=len(selected))
    def cache_progress(stage: str, rows: int, total: int, stamp: Any) -> None:
        if progress_writer is not None:
            if progress_writer.current_stage != stage:
                progress_writer.stage(stage, total_rows=total)
            progress_writer.update(rows_processed=rows, total_rows=total,
                                   last_timestamp=stamp, force=True)

    features, cache_info = load_or_build_causal_feature_cache(
        selected, cache_dir, progress=cache_progress
    )
    timing["feature_cache_and_identity_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    if progress_writer is not None:
        progress_writer.stage("benchmark_causal_context", total_rows=len(selected))
    def context_progress(event: dict[str, Any]) -> None:
        if progress_writer is not None:
            status = event.get("hmm_refit_status")
            progress_writer.update(
                rows_processed=event.get("rows_processed"),
                total_rows=event.get("total_rows"),
                last_timestamp=event.get("last_timestamp"),
                hmm_refit_status=status,
                refit_timestamp=event.get("refit_timestamp"),
                force=bool(event.get("hmm_refit_status")
                           and (str(status).endswith("in_progress") or str(status).endswith("completed"))),
            )
    full_context = CausalMarketContext()
    split_state: dict[str, Any] | None = None
    if split_rows is not None:
        if not 0 < split_rows < len(selected):
            raise ValueError("split_rows must be within the selected benchmark rows")
        full_context.bootstrap_causal_history(
            selected.iloc[:split_rows].copy(),
            precomputed_features=features.iloc[:split_rows].copy(),
            progress_callback=context_progress,
        )
        # Capture the restart point from the uninterrupted stream itself.
        # The resumed comparison then needs to replay only the suffix, rather
        # than redundantly recomputing the historical prefix a third time.
        split_state = json.loads(json.dumps(full_context.state_dict(), default=str, allow_nan=True))
        full_context.bootstrap_causal_history(
            selected.iloc[split_rows:].copy(),
            precomputed_features=features.iloc[split_rows:].copy(),
            resume=True,
            progress_callback=context_progress,
            total_rows=len(selected),
        )
    else:
        full_context.bootstrap_causal_history(selected, precomputed_features=features,
                                              progress_callback=context_progress)
    timing["uninterrupted_context_seconds"] = time.perf_counter() - tick
    result: dict[str, Any] = {
        "mode": "BOUNDED_REAL_DATA_BENCHMARK_ONLY",
        "activation_ready": False,
        "first_timestamp_utc": pd.Timestamp(selected["timestamp"].iloc[0]).isoformat(),
        "last_timestamp_utc": pd.Timestamp(selected["timestamp"].iloc[-1]).isoformat(),
        "rows": len(selected),
        "cache": cache_info,
        "timings": timing,
        "streams": {},
    }
    for name in ("mr", "s2r"):
        stream = getattr(full_context._raw_hmm_provider, name)
        result["streams"][name.upper()] = {
            "fit_count": stream.fit_count,
            "refit_events": stream.refit_events,
            "next_refit_timestamp": str(stream.next_refit_timestamp) if stream.next_refit_timestamp else None,
            "model_hash": stream.model_hash,
            "last_emitted_state": stream._last_emitted_state,
        }
    if split_rows is not None:
        assert split_state is not None
        if progress_writer is not None:
            progress_writer.stage("benchmark_resume_equivalence", total_rows=len(selected))
        tick = time.perf_counter()
        resumed = CausalMarketContext()
        resumed.load_state_dict(split_state)
        resumed.bootstrap_causal_history(
            selected.iloc[split_rows:].copy(),
            precomputed_features=features.iloc[split_rows:].copy(),
            resume=True,
            progress_callback=context_progress,
            total_rows=len(selected),
        )
        timing["split_and_restore_seconds"] = time.perf_counter() - tick
        result["resume_equivalence"] = {
            "exact_after_nonsemantic_time_fields_removed": _first_difference(
                _clean(full_context.state_dict()), _clean(resumed.state_dict())
            ) is None,
            "first_difference": _first_difference(
                _clean(full_context.state_dict()), _clean(resumed.state_dict())
            ),
            "split_rows": split_rows,
            "checkpoint_state_bytes": len(json.dumps(split_state, sort_keys=True,
                                                       separators=(",", ":"), default=str,
                                                       allow_nan=True).encode("utf-8")),
        }
    result["timings"]["total_benchmark_seconds"] = time.perf_counter() - benchmark_started
    result["timings"]["process_cpu_seconds"] = time.process_time() - cpu_started
    result["memory"] = _windows_memory_counters()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "benchmark", "start", "resume", "status"))
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--activation-utc", default="2026-10-09T00:00:00Z")
    parser.add_argument("--origin-utc", default=HISTORY_ORIGIN)
    parser.add_argument("--initial-equity", type=float, default=50_000.0)
    parser.add_argument("--chunk-rows", type=int, default=10_000)
    parser.add_argument("--benchmark-start", default="2024-07-05T00:00:00Z")
    parser.add_argument("--benchmark-end", default="2026-10-08T13:03:00Z")
    parser.add_argument("--split-rows", type=int, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coverage-acceptance", type=Path,
                        help="hash-bound Paper-only residual-data-risk acceptance artifact")
    parser.add_argument("--accept-paper-research-quality", action="store_true",
                        help="explicitly enable the recorded Paper-only residual-risk policy for this bootstrap")
    parser.add_argument("--confirm-full-bootstrap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            if not args.checkpoint.is_file():
                print(json.dumps({"available": False, "checkpoint": str(args.checkpoint)}, indent=2))
                return 2
            document = json.loads(args.checkpoint.read_text(encoding="utf-8"))
            identity = json.loads(args.identity.read_text(encoding="utf-8")) if args.identity else document.get("payload", {}).get("identity")
            from src.paper.causal_bootstrap import CausalBootstrapCheckpointStore
            payload = CausalBootstrapCheckpointStore(args.checkpoint).load(expected_identity=identity)
            out = {
                "available": True,
                "identity_source": "external_manifest" if args.identity else "self_declared_checkpoint_identity_untrusted_for_activation",
                "rows_processed": payload["rows_processed"],
                "last_processed_timestamp_utc": payload["last_processed_timestamp_utc"],
                "ready_for_activation": payload.get("ready_for_activation", False),
                "activation_timestamp_utc": payload.get("activation_timestamp_utc"),
                "coverage": payload.get("coverage_certificate"),
            }
            print(json.dumps(out, indent=2, default=str))
            return 0

        validation = validate_inputs(args.certificate)
        paper_acceptance = None
        if args.coverage_acceptance:
            if args.command in {"start", "resume"} and not args.accept_paper_research_quality:
                raise ValueError(
                    "start/resume with a non-certified certificate also requires "
                    "--accept-paper-research-quality"
                )
            from src.paper.paper_coverage_acceptance import validate_acceptance_record
            paper_acceptance = validate_acceptance_record(args.coverage_acceptance, args.certificate)
        elif args.accept_paper_research_quality:
            raise ValueError("--accept-paper-research-quality requires --coverage-acceptance")
        if args.command == "validate":
            validation["paper_coverage_acceptance"] = paper_acceptance
            print(json.dumps(validation, indent=2, default=str))
            return 0 if validation["status"] == "VALID" else 2

        if args.command == "benchmark":
            if not validation["causal_computation_possible_on_observed_bars"]:
                print(json.dumps({"started": False, "command": "benchmark",
                                  "validation": validation,
                                  "reason": "demonstrated missing required observations block benchmark processing"}, indent=2))
                return 2
            raw = _load_raw()
            from src.paper.bootstrap_artifacts import BootstrapProgressWriter
            progress_writer = BootstrapProgressWriter(args.progress)
            outcome = run_benchmark(
                raw, start=args.benchmark_start, end=args.benchmark_end,
                cache_dir=args.cache_dir, split_rows=args.split_rows,
                progress_writer=progress_writer,
            )
            progress_writer.finish()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(outcome, indent=2, default=str) + "\n", encoding="utf-8")
            print(json.dumps(outcome, indent=2, default=str))
            return 0 if outcome.get("resume_equivalence", {}).get("exact_after_nonsemantic_time_fields_removed", True) else 3

        if not args.confirm_full_bootstrap:
            raise ValueError("start/resume requires --confirm-full-bootstrap; no bootstrap was started")
        if not validation["activation_ready"] and paper_acceptance is None:
            print(json.dumps({
                "started": False,
                "command": args.command,
                "validation": validation,
                "reason": "full bootstrap is blocked until source/calendar coverage is activation-certified",
            }, indent=2, default=str))
            return 2
        if (paper_acceptance is not None
                and (not paper_acceptance.get("valid")
                     or paper_acceptance.get("policy") != "paper_research_quality_accepted")):
            raise ValueError("Paper-only coverage acceptance did not validate")
        raw = _load_raw()
        certificate_document = json.loads(args.certificate.read_text(encoding="utf-8"))
        activation = pd.Timestamp(args.activation_utc)
        origin = pd.Timestamp(args.origin_utc)
        if activation.tzinfo is None or origin.tzinfo is None:
            raise ValueError("origin and activation must be timezone-aware")
        raw = raw.loc[pd.to_datetime(raw["timestamp"], utc=True) < activation.tz_convert("UTC")].reset_index(drop=True)
        if raw.empty or pd.Timestamp(raw.timestamp.iloc[0]) != origin.tz_convert("UTC"):
            raise ValueError("causal source does not begin at the frozen origin and precede activation")
        from src.paper.bootstrap_artifacts import BootstrapProgressWriter, load_or_build_causal_feature_cache
        from src.paper.causal_bootstrap import CausalBootstrapRunner
        from src.paper.market_context import CausalMarketContext
        from src.risk.policy import XFA_50K_PRODUCTION_POLICY
        progress = BootstrapProgressWriter(args.progress)
        progress.stage("feature_cache", total_rows=len(raw))
        cache_stage = "feature_cache"
        def cache_progress(stage: str, rows: int, total: int, stamp: Any) -> None:
            nonlocal cache_stage
            if stage != cache_stage:
                progress.stage(stage, total_rows=total)
                cache_stage = stage
            progress.update(rows_processed=rows, total_rows=total,
                            last_timestamp=stamp, force=True)

        features, cache = load_or_build_causal_feature_cache(
            raw, args.cache_dir, progress=cache_progress
        )
        progress.annotate("feature_cache", cache)
        progress.stage("causal_hmm_bootstrap", total_rows=len(raw))
        def bootstrap_progress(event: dict[str, Any]) -> None:
            progress.update(
                rows_processed=event.get("rows_processed"),
                total_rows=event.get("total_rows"),
                last_timestamp=event.get("last_timestamp"),
                hmm_refit_status=event.get("hmm_refit_status"),
                refit_timestamp=event.get("refit_timestamp"),
                force=bool(event.get("checkpoint_committed")
                           or str(event.get("hmm_refit_status", "")).endswith("in_progress")
                           or str(event.get("hmm_refit_status", "")).endswith("completed")),
            )
        if paper_acceptance is not None:
            coverage_decision = paper_acceptance["decision"]
            paper_coverage_record = {
                "policy": paper_acceptance["policy"],
                "decision": coverage_decision["decision"],
                "scope": paper_acceptance["scope"],
                "acceptance_artifact_path": str(args.coverage_acceptance.resolve().relative_to(ROOT)),
                "acceptance_artifact_sha256": paper_acceptance["acceptance_artifact_sha256"],
                "strict_certificate_path": paper_acceptance["strict_certificate_path"],
                "strict_certificate_sha256": paper_acceptance["strict_certificate_sha256"],
                "strict_certificate_readiness": paper_acceptance["strict_certificate_readiness"],
                "observed_data_integrity_verified": paper_acceptance["observed_data_integrity_verified"],
                "demonstrated_missing_required_observations": paper_acceptance["demonstrated_missing_required_observations"],
                "unresolved_span_count": paper_acceptance["unresolved_span_count"],
                "unresolved_absent_minutes": paper_acceptance["unresolved_absent_minutes"],
                "accepted_degraded_dates": paper_acceptance["accepted_degraded_dates"],
                "accepted_unresolved_spans": paper_acceptance["accepted_unresolved_spans"],
            }
            coverage_validator = lambda _frame: {
                "verified": True,
                "missing_expected_minutes": 0,
                "unknown_coverage": 0,
                "certificate_readiness": validation["certificate_readiness"],
                "calendar_coverage_certificate": certificate_document,
                "paper_coverage_acceptance": paper_coverage_record,
            }
        else:
            coverage_validator = lambda _frame: {
                "verified": True, "activation_coverage_verified": True,
                "missing_expected_minutes": 0, "unknown_coverage": 0,
                "certificate_readiness": "FULLY_CERTIFIED",
                "certificate_sha256": validation["certificate_sha256"],
                "calendar_coverage_certificate": certificate_document,
            }
        runner = CausalBootstrapRunner(
            context=CausalMarketContext(), checkpoint_path=args.checkpoint,
            activation_timestamp=activation, history_origin_timestamp=origin,
            feature_cache_key=cache["cache_key"], initial_equity=args.initial_equity,
            risk_configuration=asdict(XFA_50K_PRODUCTION_POLICY),
            coverage_validator=coverage_validator,
            chunk_rows=args.chunk_rows,
        )
        payload = runner.run(raw, features, resume=args.command == "resume",
                             progress_callback=bootstrap_progress)
        identity_path = args.identity or args.checkpoint.with_suffix(".identity.json")
        _atomic_identity_manifest(identity_path, payload["identity"])
        progress.finish()
        print(json.dumps({"completed": True, "checkpoint": str(args.checkpoint),
                          "identity_manifest": str(identity_path),
                          "rows_processed": payload["rows_processed"],
                          "last_processed_timestamp_utc": payload["last_processed_timestamp_utc"],
                          "ready_for_activation": payload["ready_for_activation"]}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
