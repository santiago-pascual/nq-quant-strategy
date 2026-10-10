"""Build a conservative calendar/data-gap inventory for local MNQ OHLCV.

This tool never infers holidays. Only reviewed CME snapshots authorize a
date-specific session classification. Databento trade-derived OHLCV does not
require one row per minute; ordinary absent minutes are separated from source
degradation and never described as proof that no trade occurred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA = ROOT / "data/raw/mnq/ohlcv_1m"
DEFAULT_SNAPSHOTS = (
    ROOT / "src/paper/config/cme_mnq_calendar_2026-09-07_2026-09-08.json",
    ROOT / "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json",
)
SOURCE = "https://www.cmegroup.com/trading-hours.html"
NORMAL_SCHEDULE_RULES = [
    {
        "effective_from_local": "2012-11-18",
        "effective_until_local_exclusive": "2021-06-27T17:00:00 America/Chicago",
        "timezone": "America/Chicago",
        "equity_index_hours": "Sunday-Friday 17:00-16:15; daily pause 15:15-15:30; daily maintenance 16:15-17:00",
        "source_url": "https://www.cmegroup.com/tools-information/lookups/advisories/electronic-trading/20121022.html",
        "evidence_scope": "CME Equity Index futures recurring hours; date-specific holidays remain uncertified without reviewed snapshots.",
    },
    {
        "effective_from_local": "2021-06-27T17:00:00 America/Chicago",
        "effective_until_local_exclusive": None,
        "timezone": "America/Chicago",
        "equity_index_hours": "Sunday-Friday 17:00-16:00; daily maintenance 16:00-17:00; no daily pause",
        "source_url": "https://www.cmegroup.com/notices/electronic-trading/2021/06/20210621.html",
        "evidence_scope": "CME Equity Index contracts; date-specific holidays remain uncertified without reviewed snapshots.",
    },
]


def _snapshot_for(day: date, snapshots):
    return next((s for s in snapshots if s.coverage_start <= day <= s.coverage_end), None)


def classify_minute(stamp: pd.Timestamp, snapshots) -> str:
    """Classify schedule evidence for an absent minute, not source completeness."""
    from zoneinfo import ZoneInfo
    from src.paper.cme_calendar import NEW_YORK

    local = stamp.tz_convert(NEW_YORK)
    day = local.date()
    minute_of_day = local.hour * 60 + local.minute
    chicago = stamp.tz_convert(ZoneInfo("America/Chicago"))
    chicago_minute = chicago.hour * 60 + chicago.minute

    # Prefer a reviewed date-specific exception, including the prior evening
    # that belongs to that CME trade date.
    if minute_of_day >= 18 * 60:
        trade_date = day + timedelta(days=1)
        while trade_date.weekday() >= 5:
            trade_date += timedelta(days=1)
    elif local.weekday() == 6:
        trade_date = day + timedelta(days=1)
    else:
        trade_date = day
    snapshot = _snapshot_for(trade_date, snapshots)
    if snapshot is not None:
        from src.paper.cme_calendar import CMETradingCalendar
        calendar = CMETradingCalendar(snapshot)
        session = snapshot.session_for_rth_date(trade_date)
        if session.session_type == "closed" or session.globex_open is None or session.globex_close is None:
            return "scheduled_holiday_closure"
        local_dt = local.to_pydatetime()
        if not session.globex_open <= local_dt < session.globex_close:
            return "scheduled_holiday_or_early_close"
        if session.rth_start and session.rth_end and session.rth_start <= local_dt < session.rth_end:
            return "open_rth_minute_absent_trade_or_capture_unresolved"
        if calendar.expected_globex_minute(stamp):
            return "open_globex_minute_absent_trade_or_capture_unresolved"
        return "scheduled_closure"

    local_cutover = pd.Timestamp("2021-06-27T17:00:00", tz="America/Chicago")

    # Date-specific reviewed snapshots are authoritative for their trade date
    # and preceding Globex evening. Outside them, only recurring closures in
    # CME's published Equity Index rules are classified as closed. We do not
    # infer an open session or a holiday exception from those recurring rules.
    if chicago >= local_cutover:
        pause_start, pause_end, maintenance_start = None, None, 16 * 60
    else:
        pause_start, pause_end, maintenance_start = 15 * 60 + 15, 15 * 60 + 30, 16 * 60 + 15

    if chicago.weekday() == 5:
        return "scheduled_recurring_weekend_closure"
    if chicago.weekday() == 6 and chicago_minute < 17 * 60:
        return "scheduled_recurring_weekend_closure"
    # Friday's close starts the weekly closure; ordinary daily maintenance
    # applies only Sunday-Thursday and ends at the Sunday 17:00 CT reopen.
    if chicago.weekday() == 4 and chicago_minute >= maintenance_start:
        return "scheduled_recurring_weekend_closure"
    if chicago.weekday() < 4 and maintenance_start <= chicago_minute < 17 * 60:
        return "scheduled_recurring_daily_maintenance"
    if pause_start is not None and pause_start <= chicago_minute < pause_end:
        return "scheduled_recurring_equity_index_pause"

    return "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"


SPARSE_OHLCV_ABSENCE = "trade_derived_ohlcv_minute_absence_not_required_by_grid_contract"
DEGRADED_SOURCE_ABSENCE = "degraded_source_day_gap_unresolved"
UNKNOWN_SOURCE_ABSENCE = "source_condition_unknown_gap_unresolved"


def _load_day_conditions(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load and verify the vendor condition file included with this batch.

    Databento's condition is dataset/date scoped, not an MNQ completeness
    attestation. It is retained as provenance and used only to keep known
    degraded dates separate from ordinary sparse trade-derived OHLCV minutes.
    """
    condition_path = data_dir / "condition.json"
    manifest_path = data_dir / "manifest.json"
    if not condition_path.is_file() or not manifest_path.is_file():
        return {}, {"status": "unavailable"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest.get("files", [])
                  if item.get("filename") == condition_path.name), None)
    actual_hash = _sha256(condition_path)
    expected_hash = str(entry.get("hash", "")) if entry else ""
    if expected_hash.removeprefix("sha256:") != actual_hash:
        raise ValueError("Databento condition.json does not match its manifest hash")
    rows = json.loads(condition_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Databento condition.json must contain a list")
    conditions: dict[str, str] = {}
    for row in rows:
        day, status = str(row.get("date", "")), str(row.get("condition", ""))
        if not day or status not in {"available", "degraded"} or day in conditions:
            raise ValueError("Databento condition metadata has invalid or duplicate rows")
        conditions[day] = status
    return conditions, {"path": str(condition_path.relative_to(ROOT)),
                        "sha256": actual_hash, "bytes": condition_path.stat().st_size,
                        "manifest_path": str(manifest_path.relative_to(ROOT)),
                        "manifest_sha256": _sha256(manifest_path), "status": "verified_hash"}


def _utc_date_is_fully_scheduled_closed(day: str, snapshots) -> bool:
    start = pd.Timestamp(day, tz="UTC")
    stamps = pd.date_range(start, start + pd.Timedelta(days=1), freq="min", inclusive="left")
    closed_labels = {
        "scheduled_recurring_weekend_closure",
        "scheduled_recurring_daily_maintenance",
        "scheduled_recurring_equity_index_pause",
        "scheduled_holiday_closure",
        "scheduled_holiday_or_early_close",
        "scheduled_closure",
    }
    return all(classify_minute(stamp, snapshots) in closed_labels for stamp in stamps)


def classify_gap(previous: pd.Timestamp, current: pd.Timestamp, snapshots,
                 source_conditions: dict[str, str] | None = None) -> dict:
    if current <= previous:
        raise ValueError("gap timestamps must be strictly increasing")
    first_missing = previous.floor("min") + pd.Timedelta(minutes=1)
    end_exclusive = current.floor("min")
    total_missing = max(0, int((end_exclusive - first_missing).total_seconds() // 60))
    counts: Counter[str] = Counter()
    if total_missing:
        stamps = pd.date_range(first_missing, end_exclusive, freq="min", inclusive="left")
        from zoneinfo import ZoneInfo
        chicago = stamps.tz_convert(ZoneInfo("America/Chicago"))
        weekday = chicago.dayofweek.to_numpy()
        minute = (chicago.hour * 60 + chicago.minute).to_numpy()
        cutover = pd.Timestamp("2021-06-27T17:00:00", tz="America/Chicago").tz_convert("UTC")
        pre_rule = stamps.asi8 < cutover.value
        maintenance_start = np.where(pre_rule, 16 * 60 + 15, 16 * 60)
        labels = np.full(total_missing, "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown", dtype=object)
        weekend = (weekday == 5) | ((weekday == 6) & (minute < 17 * 60))
        friday_close = (weekday == 4) & (minute >= maintenance_start)
        maintenance = (weekday < 4) & (minute >= maintenance_start) & (minute < 17 * 60)
        pause = (weekday < 5) & (minute >= 15 * 60 + 15) & (minute < 15 * 60 + 30) & pre_rule
        labels[weekend] = "scheduled_recurring_weekend_closure"
        labels[friday_close] = "scheduled_recurring_weekend_closure"
        labels[maintenance] = "scheduled_recurring_daily_maintenance"
        labels[pause] = "scheduled_recurring_equity_index_pause"

        # Replace generic recurring classifications with reviewed date-specific
        # schedule results only for the small timestamp spans covered by each
        # explicit snapshot.
        from src.paper.cme_calendar import NEW_YORK
        for snapshot in snapshots:
            start_local = snapshot.coverage_start - timedelta(days=1)
            start_utc = pd.Timestamp(datetime.combine(start_local, time(18, 0), NEW_YORK)).tz_convert("UTC")
            end_utc = pd.Timestamp(datetime.combine(snapshot.coverage_end + timedelta(days=1), time.min, NEW_YORK)).tz_convert("UTC")
            left = max(0, int((start_utc - first_missing).total_seconds() // 60))
            right = min(total_missing, int((end_utc - first_missing).total_seconds() // 60))
            for index in range(left, max(left, right)):
                labels[index] = classify_minute(stamps[index], snapshots)

        # A minute-grid hole is not a missing required row for trade-derived
        # OHLCV: the schema emits a bar only when trades occur. Do not claim
        # that the absent interval had no trades; distinguish ordinary
        # schema-sparse rows from date-level degraded/unknown source quality.
        source_conditions = source_conditions or {}
        open_absence_labels = {
            "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown",
            "open_rth_minute_absent_trade_or_capture_unresolved",
            "open_globex_minute_absent_trade_or_capture_unresolved",
        }
        for index, label in enumerate(labels):
            if label not in open_absence_labels:
                continue
            condition = source_conditions.get(stamps[index].date().isoformat())
            if condition == "available":
                labels[index] = SPARSE_OHLCV_ABSENCE
            elif condition == "degraded":
                labels[index] = DEGRADED_SOURCE_ABSENCE
            else:
                labels[index] = UNKNOWN_SOURCE_ABSENCE
        counts.update(labels.tolist())
    if len(counts) == 0:
        status = "no_missing_minute"
    elif any("unresolved" in key for key in counts):
        status = "unresolved_source_quality_or_calendar"
    elif SPARSE_OHLCV_ABSENCE in counts:
        status = ("sparse_trade_derived_ohlcv_absence"
                  if len(counts) == 1 else "mixed_sparse_ohlcv_and_scheduled_closure")
    else:
        status = "verified_scheduled_closure"
    return {
        "previous_observed_utc": previous.isoformat(),
        "next_observed_utc": current.isoformat(),
        "elapsed_minutes": int((current - previous).total_seconds() // 60),
        "missing_minutes": total_missing,
        "classification": status,
        "minute_class_counts": dict(sorted(counts.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_certificate(data_dir: Path, snapshot_paths: Iterable[Path]) -> tuple[dict, list[dict]]:
    from src.paper.cme_calendar import CMECalendarSnapshot

    snapshots = [CMECalendarSnapshot.from_json(path) for path in snapshot_paths]
    source_conditions, condition_provenance = _load_day_conditions(data_dir)
    files = sorted(data_dir.glob("*.csv.zst"))
    if not files:
        raise FileNotFoundError(f"No .csv.zst input files in {data_dir}")
    previous = None
    first = last = None
    rows = 0
    observed_degraded_rows: Counter[str] = Counter()
    duplicate_count = 0
    nonmonotonic_count = 0
    gaps = []
    for path in files:
        for chunk in pd.read_csv(path, compression="zstd", usecols=["ts_event"], chunksize=250_000):
            stamps = pd.to_datetime(chunk["ts_event"], unit="ns", utc=True, errors="raise")
            if stamps.isna().any():
                raise ValueError(f"Null timestamp found in {path.name}")
            if len(stamps) > 1:
                duplicate_count += int(stamps.duplicated().sum())
                nonmonotonic_count += int((stamps.diff().dropna() <= pd.Timedelta(0)).sum())
            for stamp in stamps:
                stamp = pd.Timestamp(stamp)
                if source_conditions.get(stamp.date().isoformat()) == "degraded":
                    observed_degraded_rows[stamp.date().isoformat()] += 1
                if first is None:
                    first = stamp
                if previous is not None:
                    delta = stamp - previous
                    if delta <= pd.Timedelta(0):
                        if delta == pd.Timedelta(0):
                            duplicate_count += 1
                        else:
                            nonmonotonic_count += 1
                    elif delta > pd.Timedelta(minutes=1):
                        gaps.append(classify_gap(previous, stamp, snapshots, source_conditions))
                previous = stamp
                last = stamp
                rows += 1
    classes = Counter(gap["classification"] for gap in gaps)
    minutes = Counter()
    for gap in gaps:
        minutes.update(gap["minute_class_counts"])
    first_day = pd.Timestamp(first).date().isoformat() if first is not None else None
    last_day = pd.Timestamp(last).date().isoformat() if last is not None else None
    all_degraded_source_dates = sorted(day for day, condition in source_conditions.items()
                                       if condition == "degraded" and first_day and last_day
                                       and first_day <= day <= last_day)
    degraded_dates_with_rows = sorted(day for day in all_degraded_source_dates
                                      if observed_degraded_rows.get(day, 0) > 0)
    degraded_dates_without_rows = sorted(day for day in all_degraded_source_dates
                                         if observed_degraded_rows.get(day, 0) == 0)
    degraded_zero_row_dates_verified_closed = sorted(
        day for day in degraded_dates_without_rows
        if _utc_date_is_fully_scheduled_closed(day, snapshots)
    )
    blocking_degraded_dates = sorted(set(degraded_dates_with_rows)
                                     | (set(degraded_dates_without_rows)
                                        - set(degraded_zero_row_dates_verified_closed)))
    snapshot_records = []
    for path, snap in zip(snapshot_paths, snapshots):
        review_path = path.with_name(path.stem + ".review.json")
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
        snapshot_records.append({
            "path": str(path.relative_to(ROOT)),
            "review_path": str(review_path.relative_to(ROOT)) if review_path.is_file() else None,
            "review_status": review.get("review_status"),
            "reviewed_at_utc": review.get("reviewed_at_utc"),
            "source_url": review.get("source_url", snap.source),
            "version": snap.version,
            "coverage_start": snap.coverage_start.isoformat(),
            "coverage_end": snap.coverage_end.isoformat(),
            "identity": snap.identity,
            "exceptions": {key: dict(value) for key, value in snap.exceptions.items()},
        })
    body = {
        "schema_version": 2,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "calendar_source": SOURCE,
        "normal_schedule_rules": NORMAL_SCHEDULE_RULES,
        "calendar_snapshots": snapshot_records,
        "data_files": [{"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
                       for path in files],
        "source_quality_metadata": {
            **condition_provenance,
            "scope": "Databento dataset/date condition; not instrument-level completeness evidence",
            "condition_dates_in_observed_range": sum(
                1 for day in source_conditions if first_day and last_day and first_day <= day <= last_day
            ),
            "degraded_dates_in_observed_range": blocking_degraded_dates,
            "degraded_date_count_in_observed_range": len(blocking_degraded_dates),
            "all_degraded_condition_dates_in_observed_range": all_degraded_source_dates,
            "degraded_dates_with_observed_mnq_rows": degraded_dates_with_rows,
            "observed_bar_count_by_degraded_date": {
                day: observed_degraded_rows.get(day, 0) for day in all_degraded_source_dates
            },
            "degraded_dates_without_observed_rows": degraded_dates_without_rows,
            "zero_row_degraded_dates_verified_fully_closed": degraded_zero_row_dates_verified_closed,
        },
        "input": {"rows": rows, "first_utc": first.isoformat() if first is not None else None,
                  "last_utc": last.isoformat() if last is not None else None,
                  "duplicate_timestamps": duplicate_count,
                  "nonmonotonic_transitions": nonmonotonic_count},
        "gap_count": len(gaps),
        "gap_classification_counts": dict(sorted(classes.items())),
        "missing_minute_classification_counts": dict(sorted(minutes.items())),
        "policy": {
            "regular_schedule": "Classify only CME-published recurring Equity Index closures outside reviewed snapshots; no date-specific holiday hours are inferred.",
            "OHLCV_absence": "Databento trade-derived OHLCV need not emit a row for every minute. An absent minute on a dataset-available date is classified as schema-sparse and is not a demonstrated missing required observation; it is not proof that no trade occurred. Date-level degraded or missing condition evidence remains unresolved.",
            "minute_grid_required": False,
            "uncovered_dates": "Historical date-specific holiday hours are not certified by this inventory. Sparse OHLCV absences do not certify holiday schedules; source-degraded and source-condition-unknown intervals remain unresolved.",
            "synthetic_bars": False,
        },
        "readiness": "NOT_FULLY_CERTIFIED" if blocking_degraded_dates or any(
            "unresolved" in key for key in minutes
        ) else "SPARSE_TRADE_OBSERVATIONS_VALIDATED_CALENDAR_PARTIAL",
    }
    return body, gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--snapshot", type=Path, action="append", default=None,
                        help="Reviewed runtime snapshot; repeat for multiple windows.")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON certificate output path; gap CSV is written beside it.")
    args = parser.parse_args()
    paths = args.snapshot or list(DEFAULT_SNAPSHOTS)
    certificate, gaps = build_certificate(args.data_dir, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output.with_name(args.output.stem + "_gaps.csv")
    args.output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(gaps).to_csv(detail_path, index=False)
    print(json.dumps({"certificate": str(args.output), "gap_detail": str(detail_path),
                      "rows": certificate["input"]["rows"], "gaps": certificate["gap_count"],
                      "classification": certificate["gap_classification_counts"],
                      "readiness": certificate["readiness"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
