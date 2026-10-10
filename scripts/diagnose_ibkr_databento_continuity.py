"""Bounded read-only IBKR/Dataset continuity audit for 2026-10-08.

Requests the open-session gap in <=30-minute historical TRADES slices, followed
by a 30-minute Databento overlap. It does not modify market data or Paper state.
Run with the Paper Python environment and the isolated ibapi site-packages on
PYTHONPATH.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any
from uuid import uuid4

import pandas as pd

from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.ibkr_delayed_acquisition import (
    HistoricalRequest,
    HistoricalRequestPacer,
)
from src.paper.ibkr_delayed_diagnostic import validate_historical_bars
from src.paper.ibkr_paper_runner import TWSHistoricalTRADESClient


UTC = timezone.utc
CON_ID = 815824267
LOCAL_SYMBOL = "MNQZ6"
EXPIRY = "20261218"
DATASET = "data/raw/mnq/ohlcv_1m/glbx-mdp3-20261008-20261008T1302.ohlcv-1m.csv.zst"
CALENDAR = "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json"


def _utc(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, UTC)


def _chunks(timestamps: list[datetime], maximum: int = 30) -> list[list[datetime]]:
    chunks: list[list[datetime]] = []
    for stamp in timestamps:
        if (not chunks or len(chunks[-1]) >= maximum
                or stamp - chunks[-1][-1] != timedelta(minutes=1)):
            chunks.append([stamp])
        else:
            chunks[-1].append(stamp)
    return chunks


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def _json_safe(value: Any) -> Any:
    """Keep malformed provider values auditable without non-standard JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    output = Path("results/paper/ibkr_diagnostics") / f"continuity_audit_20261008_{uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=False)
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(CALENDAR))

    databento = pd.read_csv(DATASET, compression="zstd",
                            usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol", "instrument_id"])
    db_stamps = pd.to_datetime(databento["ts_event"], unit="ns", utc=True)
    if databento.empty or databento["symbol"].astype(str).ne("MNQ.v.0").any():
        raise RuntimeError("Databento overlap source is empty or not MNQ.v.0")
    databento["timestamp_utc"] = db_stamps
    for name in ("open", "high", "low", "close"):
        databento[name] = pd.to_numeric(databento[name], errors="raise") / 1_000_000_000
    if db_stamps.duplicated().any() or not db_stamps.is_monotonic_increasing:
        raise RuntimeError("Databento partial partition has duplicate or unordered timestamps")
    db_last = db_stamps.max().to_pydatetime()
    if db_last != datetime(2026, 10, 8, 13, 2, tzinfo=UTC):
        raise RuntimeError(f"Unexpected Databento last bar: {db_last.isoformat()}")

    recovery_start = datetime(2026, 10, 8, 13, 3, tzinfo=UTC)
    recovery_end_exclusive = datetime(2026, 10, 8, 23, 48, tzinfo=UTC)
    expected = calendar.expected_missing_minutes(db_last, recovery_end_exclusive)
    expected = [stamp for stamp in expected if recovery_start <= stamp < recovery_end_exclusive]
    if not expected or expected[0] != recovery_start or expected[-1] != datetime(2026, 10, 8, 23, 47, tzinfo=UTC):
        raise RuntimeError("Reviewed CME calendar did not produce the expected bounded gap endpoints")

    overlap_start = datetime(2026, 10, 8, 12, 33, tzinfo=UTC)
    overlap_end_exclusive = datetime(2026, 10, 8, 13, 3, tzinfo=UTC)
    overlap_expected = [overlap_start + timedelta(minutes=i) for i in range(30)]
    requests_to_make: list[dict[str, Any]] = []
    for chunk in _chunks(expected):
        requests_to_make.append({"kind": "gap_recovery", "expected": chunk})
    requests_to_make.append({"kind": "databento_overlap", "expected": overlap_expected})
    if len(requests_to_make) > 40:
        raise RuntimeError("bounded request plan exceeds conservative 10-minute request budget")

    client = TWSHistoricalTRADESClient(
        host="127.0.0.1", port=7496, client_id=86,
        con_id=CON_ID, local_symbol=LOCAL_SYMBOL, expiry=EXPIRY,
        timeout_seconds=20,
    )
    contract = client.contract
    contract_record = {
        "symbol": contract.symbol, "sec_type": contract.secType,
        "local_symbol": contract.localSymbol, "con_id": int(contract.conId),
        "expiry": str(contract.lastTradeDateOrContractMonth),
        "exchange": contract.exchange, "currency": contract.currency,
        "multiplier": str(contract.multiplier),
    }
    if (contract_record["con_id"], contract_record["local_symbol"], contract_record["expiry"]) != (CON_ID, LOCAL_SYMBOL, EXPIRY):
        client.close()
        raise RuntimeError(f"IBKR contract identity mismatch: {contract_record}")

    pacer = HistoricalRequestPacer(minimum_cycle_seconds=30, max_requests_per_10_minutes=40)
    raw_observations: list[dict[str, Any]] = []
    request_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    last_request_epoch: float | None = None
    try:
        for index, spec in enumerate(requests_to_make, start=1):
            print(f"IBKR bounded request {index}/{len(requests_to_make)} ({spec['kind']})", flush=True)
            if last_request_epoch is not None:
                wait = 30 - (time.time() - last_request_epoch)
                if wait > 0:
                    time.sleep(wait)
            now = datetime.now(UTC)
            pacer.begin_cycle(now, 1)
            expected_rows = spec["expected"]
            start, last = expected_rows[0], expected_rows[-1]
            end_exclusive = last + timedelta(minutes=1)
            seconds = int((end_exclusive - start).total_seconds())
            request = HistoricalRequest(
                "pending_confirmation", seconds, end_exclusive,
                covered_pending_starts=tuple(int(stamp.timestamp()) for stamp in expected_rows),
            )
            request_id = 200 + index
            response = client.request(request, request_id)
            observed_at = datetime.now(UTC).isoformat()
            last_request_epoch = time.time()
            rows = list(response.get("rows", []))
            for row in rows:
                raw_observations.append({
                    **row,
                    "request_id": request_id,
                    "request_kind": spec["kind"],
                    "requested_start_utc": start.isoformat(),
                    "requested_end_exclusive_utc": end_exclusive.isoformat(),
                    "response_observed_at_utc": observed_at,
                    "source": "IBKR TWS delayed historical TRADES; demo provenance unverified",
                })
            expected_set = {int(stamp.timestamp()) for stamp in expected_rows}
            result_stamps = [int(row["timestamp"]) for row in rows]
            returned_in_bounds = [row for row in rows if int(row["timestamp"]) in expected_set]
            request_summary = {
                "request_id": request_id, "kind": spec["kind"],
                "requested_start_utc": start.isoformat(),
                "requested_end_exclusive_utc": end_exclusive.isoformat(),
                "requested_minutes": len(expected_rows), "duration_seconds": seconds,
                "completed": bool(response.get("completed")),
                "connected": bool(response.get("connected")),
                "returned_rows": len(rows), "returned_expected_minutes": len(returned_in_bounds),
                "first_returned_utc": _utc(min(result_stamps)).isoformat() if result_stamps else None,
                "last_returned_utc": _utc(max(result_stamps)).isoformat() if result_stamps else None,
                "response_observed_at_utc": observed_at,
                "errors": list(response.get("errors", [])),
            }
            request_summaries.append(request_summary)
            if not response.get("completed") or not response.get("connected"):
                failures.append(request_summary)
    finally:
        client.close()

    # Append every callback observation, including repeated timestamps/versions.
    observations_path = output / "ibkr_raw_observations.jsonl"
    with observations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in raw_observations:
            handle.write(json.dumps(_json_safe(row), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()

    recovery_expected = {int(stamp.timestamp()) for stamp in expected}
    recovery_observations = [row for row in raw_observations
                             if row["request_kind"] == "gap_recovery"
                             and int(row["timestamp"]) in recovery_expected]
    by_stamp: dict[int, list[dict[str, Any]]] = {}
    for row in recovery_observations:
        by_stamp.setdefault(int(row["timestamp"]), []).append(row)
    recovered = sorted(by_stamp)
    missing_epochs = sorted(recovery_expected - set(recovered))
    duplicates = {stamp: rows for stamp, rows in by_stamp.items() if len(rows) > 1}
    revision_conflicts = {
        stamp: rows for stamp, rows in duplicates.items()
        if len({tuple(float(row[k]) for k in ("open", "high", "low", "close", "volume")) for row in rows}) > 1
    }
    canonical_recovery: list[dict[str, Any]] = []
    for stamp in recovered:
        versions = by_stamp[stamp]
        values = {tuple(float(row[k]) for k in ("open", "high", "low", "close", "volume")) for row in versions}
        if len(values) == 1:
            canonical_recovery.append(versions[-1])
    quality = validate_historical_bars(canonical_recovery)
    recovered_timestamps = [_utc(stamp) for stamp in recovered]
    calendar_unexpected = [stamp.isoformat() for stamp in recovered_timestamps
                           if not calendar.expected_globex_minute(pd.Timestamp(stamp))]

    db_by_stamp = {
        int(row.timestamp_utc.timestamp()): row
        for row in databento.loc[databento["timestamp_utc"].between(overlap_start, overlap_end_exclusive - timedelta(minutes=1))].itertuples()
    }
    overlap_rows = [row for row in raw_observations if row["request_kind"] == "databento_overlap"
                    and int(row["timestamp"]) in db_by_stamp]
    overlap_unique: dict[int, dict[str, Any]] = {}
    for row in overlap_rows:
        overlap_unique.setdefault(int(row["timestamp"]), row)
    differences: dict[str, list[float]] = {key: [] for key in ("open", "high", "low", "close", "volume")}
    match_records: list[dict[str, Any]] = []
    overlap_invalid: list[str] = []
    for stamp, row in sorted(overlap_unique.items()):
        db = db_by_stamp[stamp]
        try:
            ibkr_values = {field: float(row[field]) for field in differences}
            if not all(math.isfinite(value) for value in ibkr_values.values()):
                raise ValueError("non-finite IBKR OHLCV")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            overlap_invalid.append(f"{_utc(stamp).isoformat()}: {exc}")
            continue
        rec: dict[str, Any] = {"timestamp_utc": _utc(stamp).isoformat(),
                               "databento_instrument_id": int(db.instrument_id)}
        for field in differences:
            a, b = float(getattr(db, field)), ibkr_values[field]
            delta = b - a
            differences[field].append(delta)
            rec[f"databento_{field}"] = a
            rec[f"ibkr_{field}"] = b
            rec[f"{field}_difference_ibkr_minus_databento"] = delta
        rec["ibkr_contract_id"] = int(row["con_id"])
        match_records.append(rec)

    recovery_rows_path = output / "ibkr_recovered_expected_bars.csv"
    with recovery_rows_path.open("w", encoding="utf-8", newline="") as handle:
        columns = ["timestamp_utc", "timestamp_epoch", "con_id", "local_symbol", "open", "high", "low", "close", "volume", "arrival_utc", "request_id", "response_observed_at_utc"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in canonical_recovery:
            writer.writerow({"timestamp_utc": _utc(int(row["timestamp"])).isoformat(),
                             "timestamp_epoch": int(row["timestamp"]), "con_id": int(row["con_id"]),
                             "local_symbol": LOCAL_SYMBOL,
                             **{key: row[key] for key in ("open", "high", "low", "close", "volume")},
                             "arrival_utc": row.get("arrival_utc"), "request_id": row.get("request_id"),
                             "response_observed_at_utc": row.get("response_observed_at_utc")})
    overlap_path = output / "databento_ibkr_overlap.csv"
    pd.DataFrame(match_records).to_csv(overlap_path, index=False)

    missing_times = [_utc(stamp).isoformat() for stamp in missing_epochs]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True, "orders_submitted": False, "paid_data_download": False,
        "provider_provenance": "TWS Demo as stated by operator; not cryptographically verified",
        "contract": contract_record,
        "request_policy": {"bar_size": "1 min", "what_to_show": "TRADES", "use_rth": False,
                           "format_date": 2, "max_request_seconds": 1800,
                           "minimum_start_spacing_seconds": 30,
                           "rolling_request_budget_per_10_minutes": 40},
        "databento_source": {"path": DATASET, "symbol": "MNQ.v.0", "first_utc": db_stamps.min().isoformat(),
                             "last_utc": db_last.isoformat(), "instrument_ids": sorted(map(int, databento.instrument_id.unique())),
                             "price_scale_applied": "raw integer price divided by 1e9; no further normalization"},
        "gap": {"last_databento_bar_utc": db_last.isoformat(),
                "first_prior_ibkr_sample_bar_utc": "2026-10-08T23:48:00+00:00",
                "requested_interval_start_utc": recovery_start.isoformat(),
                "requested_interval_end_utc_inclusive": "2026-10-08T23:47:00+00:00",
                "maintenance_break_utc": ["2026-10-08T21:00:00+00:00", "2026-10-08T21:59:00+00:00"],
                "expected_open_minutes": len(expected), "recovered_unique_expected_minutes": len(recovered),
                "still_missing_expected_minutes": len(missing_epochs), "missing_timestamps_utc": missing_times,
                "duplicate_timestamp_count": len(recovery_observations) - len(recovered),
                "conflicting_revision_timestamps": [_utc(stamp).isoformat() for stamp in sorted(revision_conflicts)],
                "earliest_recovered_utc": _utc(recovered[0]).isoformat() if recovered else None,
                "latest_recovered_utc": _utc(recovered[-1]).isoformat() if recovered else None,
                "bar_quality": quality, "calendar_unexpected_rows": calendar_unexpected},
        "requests": request_summaries,
        "provider_failures": failures,
        "overlap": {"requested_start_utc": overlap_start.isoformat(),
                    "requested_end_utc_inclusive": "2026-10-08T13:02:00+00:00",
                    "matched_timestamp_count": len(match_records),
                    "missing_databento_timestamps": [stamp.isoformat() for stamp in overlap_expected
                        if int(stamp.timestamp()) not in db_by_stamp],
                    "missing_ibkr_timestamps": [_utc(stamp).isoformat() for stamp in
                        sorted({int(s.timestamp()) for s in overlap_expected} - set(overlap_unique))],
                    "difference_statistics": {field: _stats(vals) for field, vals in differences.items()},
                    "invalid_ibkr_rows": overlap_invalid,
                    "ohlc_exact_match_count": sum(all(abs(float(r[f"{k}_difference_ibkr_minus_databento"])) < 1e-9 for k in ("open", "high", "low", "close")) for r in match_records),
                    "volume_exact_match_count": sum(abs(float(r["volume_difference_ibkr_minus_databento"])) < 1e-9 for r in match_records),
                    "comparison_rows": str(overlap_path)},
        "contract_mapping_assessment": {
            "databento_symbol": "MNQ.v.0", "databento_instrument_id_at_overlap": 42005282,
            "ibkr_contract": contract_record,
            "local_crosswalk_available": False,
            "evidence_limit": "Databento local manifest records continuous input/output modes and bar instrument_id, but no raw-symbol/expiry definition crosswalk. Price overlap can support a comparison but cannot prove instrument identity or continuous-roll equivalence.",
        },
        "artifacts": {"raw_observations_jsonl": str(observations_path),
                      "recovered_expected_bars_csv": str(recovery_rows_path),
                      "overlap_csv": str(overlap_path), "report_json": str(output / "report.json")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "gap": report["gap"],
                      "overlap": {key: value for key, value in report["overlap"].items() if key != "comparison_rows"},
                      "contract_mapping_assessment": report["contract_mapping_assessment"],
                      "provider_failures": len(failures)}, indent=2, sort_keys=True))
    return 0 if not failures and not missing_epochs and not revision_conflicts else 2


if __name__ == "__main__":
    raise SystemExit(main())
