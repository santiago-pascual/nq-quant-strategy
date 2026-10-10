"""Bounded, read-only IBKR historical-bar polling feasibility diagnostic.

This utility is not a Paper MarketDataSource and has no order API calls.
Historical response arrival age is recorded, but is not represented as true
live-feed publication latency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import csv
import json
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any

from src.paper.ibkr_delayed_diagnostic import validate_historical_bars
from src.paper.ibkr_observation_ledger import AppendOnlyBarObservationLedger, load_observation_ledger
from src.paper.cme_calendar import CalendarUnavailable, CMECalendarSnapshot, CMETradingCalendar
from src.paper.ibkr_delayed_market_data import (
    AppendOnlyFinalizationLedger,
    FinalizationPolicy,
    IBKRContractSchedule,
    IBKRContractWindow,
    finalize_observation_sequence,
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def analyze_poll_snapshots(snapshots: list[dict[str, Any]], *, interval_seconds: float,
                           calendar: CMETradingCalendar | None = None) -> dict[str, Any]:
    """Deduplicate, assess completed bars and compare overlapping snapshots."""
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    revisions: list[dict[str, Any]] = []
    new_per_poll: list[int] = []
    arrival_lags: list[float] = []
    errors: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    duplicate_rows = 0
    out_of_order = 0

    for poll in snapshots:
        poll_time = float(poll["request_epoch"])
        errors.extend(poll.get("errors", []))
        rows = poll.get("bars", [])
        observed: set[tuple[int, int]] = set()
        raw_stamps: list[int] = []
        newly_seen = 0
        for row in rows:
            try:
                con_id = int(row["con_id"])
                stamp = int(row["timestamp"])
                values = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError("non-finite OHLCV")
                if (min(values[k] for k in ("open", "high", "low", "close")) <= 0 or
                        values["volume"] < 0 or values["high"] < max(values["open"], values["close"], values["low"]) or
                        values["low"] > min(values["open"], values["close"], values["high"])):
                    raise ValueError("invalid OHLCV bounds")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                invalid.append({"poll": poll["poll"], "error": str(exc), "row": row})
                continue
            raw_stamps.append(stamp)
            arrived = row.get("arrival_utc") or poll.get("arrival_utc") or _iso_now()
            try:
                arrival_epoch = datetime.fromisoformat(arrived).timestamp()
            except (TypeError, ValueError):
                arrival_epoch = poll_time
            # Completion is judged at actual client observation time, not at
            # request submission time. Never expose a still-forming minute.
            if stamp + 60 > arrival_epoch:
                continue
            key = (con_id, stamp)
            if key in observed:
                duplicate_rows += 1
            observed.add(key)
            arrival_lags.append(max(0.0, arrival_epoch - (stamp + 60)))
            if key in unique:
                duplicate_rows += 1
            if key not in unique:
                unique[key] = {"con_id": con_id, "timestamp": stamp, **values, "first_seen_utc": arrived,
                               "last_seen_utc": arrived, "first_seen_poll": poll["poll"]}
                newly_seen += 1
            else:
                prior = unique[key]
                changed = {field: {"old": prior[field], "new": values[field]}
                           for field in values if prior[field] != values[field]}
                if changed:
                    revisions.append({"con_id": con_id, "timestamp": stamp,
                                      "timestamp_utc": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                                      "poll": poll["poll"], "changes": changed})
                    prior.update(values)
                prior["last_seen_utc"] = arrived
        new_per_poll.append(newly_seen)
        out_of_order += sum(1 for left, right in zip(raw_stamps, raw_stamps[1:]) if left > right)

    ordered = sorted(unique.values(), key=lambda item: (item["con_id"], item["timestamp"]))
    # Preserve exact OHLCV validator semantics while reporting gaps separately.
    quality = validate_historical_bars(ordered)
    gaps = []
    for left, right in zip(ordered, ordered[1:]):
        if left["con_id"] == right["con_id"] and right["timestamp"] - left["timestamp"] > 60:
            missing_minutes = (right["timestamp"] - left["timestamp"]) // 60 - 1
            missing_stamps = [left["timestamp"] + 60 * offset for offset in range(1, int(missing_minutes) + 1)]
            expected_count: int | None = None
            if calendar is None:
                classification = "unclassified_market_gap; calendar not supplied"
            else:
                try:
                    expected_count = sum(
                        calendar.expected_globex_minute(datetime.fromtimestamp(stamp, timezone.utc))
                        for stamp in missing_stamps
                    )
                    if expected_count == 0:
                        classification = "verified_scheduled_closure"
                    elif expected_count == missing_minutes:
                        classification = "expected_session_gap_unresolved; OHLCV absence does not prove data loss"
                    else:
                        classification = "mixed_closure_and_expected_session_minutes_unresolved"
                except (CalendarUnavailable, KeyError, ValueError) as exc:
                    classification = f"uncertified_calendar_gap: {type(exc).__name__}: {exc}"
                    expected_count = None
            gaps.append({"con_id": left["con_id"], "after_utc": datetime.fromtimestamp(left["timestamp"], timezone.utc).isoformat(),
                         "before_utc": datetime.fromtimestamp(right["timestamp"], timezone.utc).isoformat(),
                         "missing_minute_slots": missing_minutes,
                         "expected_open_minute_slots": expected_count,
                         "classification": classification})
    timestamps = [item["timestamp"] for item in ordered]
    return {
        "poll_count": len(snapshots), "poll_interval_seconds": interval_seconds,
        "completed_unique_bar_count": len(ordered), "new_completed_bars_by_poll": new_per_poll,
        "new_bar_available_during_test": any(count > 0 for count in new_per_poll[1:]),
        "new_bar_count_after_first_poll": sum(new_per_poll[1:]),
        "overlap_revision_count": len(revisions), "revisions": revisions,
        "bar_quality": quality, "invalid_rows": invalid,
        "timestamp_gaps": gaps, "gap_count": len(gaps),
        "duplicate_rows_across_responses": duplicate_rows,
        "out_of_order_timestamp_count": out_of_order,
        "retrieval_age_seconds": {
            "min": min(arrival_lags) if arrival_lags else None,
            "median": sorted(arrival_lags)[len(arrival_lags) // 2] if arrival_lags else None,
            "max": max(arrival_lags) if arrival_lags else None,
            "interpretation": "request-time minus completed bar end; historical retrieval age, not true live publication latency",
        },
        "provider_errors": errors,
        "errors": errors,
        "completed_bars": ordered,
    }


def _poll_verdict(outcome: str, analysis: dict[str, Any],
                 finalization_smoke: dict[str, Any] | None) -> str:
    """Do not treat a safely withheld pre-finalization revision as feed failure."""
    if (outcome != "completed" or not analysis["completed_unique_bar_count"]
            or analysis["invalid_rows"]):
        return "historical_poll_not_validated"
    if not analysis["overlap_revision_count"]:
        return "historical_poll_usable_for_further_paper_development"
    if (finalization_smoke
            and finalization_smoke.get("finalized_bars", 0) > 0
            and not finalization_smoke.get("late_revisions")
            and not finalization_smoke.get("error")):
        return "historical_poll_usable_after_revision_stability_gate"
    return "historical_poll_not_validated"


def run_poll_test(*, host: str = "127.0.0.1", port: int = 7496, client_id: int = 74,
                  con_id: int = 815824267, local_symbol: str = "MNQZ6", polls: int = 3,
                  interval_seconds: float = 30.0, duration_seconds: int = 1800,
                  timeout: float = 15.0, output_dir: str | Path = "results/paper/ibkr_diagnostics/historical_poll",
                  cme_calendar_snapshot: str | Path | None = None) -> dict[str, Any]:
    if polls < 1 or polls > 5:
        raise ValueError("polls must be between 1 and 5")
    if interval_seconds < 15:
        raise ValueError("poll interval must be at least 15 seconds for identical historical requests")
    if duration_seconds < 60 or duration_seconds > 86400:
        raise ValueError("duration_seconds must be between 60 and 86400")
    endpoint = f"{host}:{port}"
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(cme_calendar_snapshot)) if cme_calendar_snapshot else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ledger_path = target / f"ibkr_bar_observations_{run_id}.jsonl"
    ledger = AppendOnlyBarObservationLedger(ledger_path, run_id=run_id)
    try:
        with socket.create_connection((host, port), timeout=2):
            reachable = True
    except OSError as exc:
        return {"verdict": "unavailable", "endpoint": endpoint, "tws_reachable": False,
                "provenance": "TWS Demo (unverified by this diagnostic)", "read_only": True,
                "orders_submitted": False, "error": str(exc), "polls": []}

    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as exc:
        return {"verdict": "dependency_missing", "endpoint": endpoint, "tws_reachable": reachable,
                "provenance": "TWS Demo (unverified by this diagnostic)", "read_only": True,
                "orders_submitted": False, "error": str(exc), "polls": []}

    class PollApp(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self.connected = threading.Event()
            self.contract_done = threading.Event()
            self.poll_done = threading.Event()
            self.contracts: list[Any] = []
            self.active_bars: list[dict[str, Any]] = []
            self.poll_errors: list[dict[str, Any]] = []
            self.all_errors: list[dict[str, Any]] = []
            self.active_req: int | None = None
            self.current_contract: dict[str, Any] = {"con_id": con_id, "local_symbol": local_symbol}
            self.current_poll_number = 0

        def nextValidId(self, orderId: int) -> None:
            self.connected.set()

        def contractDetails(self, reqId: int, details: Any) -> None:
            if reqId == 1:
                self.contracts.append(details)

        def contractDetailsEnd(self, reqId: int) -> None:
            if reqId == 1:
                self.contract_done.set()

        def historicalData(self, reqId: int, bar: Any) -> None:
            if reqId != self.active_req:
                return
            try:
                stamp = int(bar.date)
                row = {"con_id": con_id, "timestamp": stamp, "open": float(bar.open), "high": float(bar.high),
                       "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume),
                       "arrival_utc": _iso_now()}
                ledger.append(row, contract=self.current_contract, observed_at=row["arrival_utc"],
                              request_id=reqId, poll_number=self.current_poll_number)
                self.active_bars.append(row)
            except (TypeError, ValueError, OverflowError) as exc:
                self.poll_errors.append({"request_id": reqId, "message": f"invalid historical bar: {exc}"})

        def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
            if reqId == self.active_req:
                self.poll_done.set()

        def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:
            error = {"request_id": int(reqId), "code": int(errorCode), "message": str(errorString), "arrival_utc": _iso_now()}
            self.all_errors.append(error)
            if reqId == self.active_req:
                self.poll_errors.append(error)
                # IB errors indicating a terminal historical response.
                if int(errorCode) in {162, 200, 321, 366}:
                    self.poll_done.set()

    app = PollApp()
    reader: threading.Thread | None = None
    snapshots: list[dict[str, Any]] = []
    contract_result: dict[str, Any] = {"con_id": con_id, "local_symbol": local_symbol}
    outcome = "failed"
    failure: str | None = None
    api_connected = False
    contract_resolved = False
    started = time.monotonic()
    started_utc = _iso_now()
    try:
        app.connect(host, port, client_id)
        reader = threading.Thread(target=app.run, name="ibkr-historical-poll", daemon=True)
        reader.start()
        if not app.connected.wait(timeout):
            raise RuntimeError("IBKR API handshake timed out")
        api_connected = True
        query = Contract()
        query.conId = con_id
        query.exchange = "CME"
        app.reqContractDetails(1, query)
        if not app.contract_done.wait(timeout):
            raise RuntimeError("contract identity lookup timed out")
        matches = [d.contract for d in app.contracts if int(d.contract.conId) == con_id]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one contract for conId {con_id}; got {len(matches)}")
        contract = matches[0]
        if contract.symbol != "MNQ" or contract.secType != "FUT" or contract.localSymbol != local_symbol:
            raise RuntimeError(f"resolved contract identity mismatch: {contract.localSymbol=} {contract.symbol=} {contract.secType=}")
        contract_resolved = True
        contract_result = {"symbol": contract.symbol, "local_symbol": contract.localSymbol,
                           "con_id": int(contract.conId), "expiry": contract.lastTradeDateOrContractMonth,
                           "exchange": contract.exchange, "currency": contract.currency,
                           "multiplier": contract.multiplier}
        app.current_contract = {"con_id": int(contract.conId), "local_symbol": contract.localSymbol,
                                "expiry": contract.lastTradeDateOrContractMonth}
        app.reqMarketDataType(3)

        for poll_index in range(polls):
            if poll_index:
                time.sleep(interval_seconds)
            now = datetime.now(timezone.utc)
            request_epoch = now.timestamp()
            app.active_bars = []
            app.poll_errors = []
            app.poll_done.clear()
            req_id = 100 + poll_index
            app.active_req = req_id
            app.current_poll_number = poll_index + 1
            app.reqHistoricalData(req_id, contract, "", f"{duration_seconds} S", "1 min", "TRADES", 0, 2, False, [])
            completed = app.poll_done.wait(timeout * 4)
            snapshots.append({"poll": poll_index + 1, "request_utc": now.isoformat(), "request_epoch": request_epoch,
                              "arrival_utc": _iso_now(), "request_completed": completed,
                              "bars": list(app.active_bars), "errors": list(app.poll_errors)})
            if not completed:
                failure = f"poll {poll_index + 1} timed out"
                break
        outcome = "completed" if snapshots and all(item["request_completed"] for item in snapshots) else "incomplete"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if app.isConnected():
            app.disconnect()
        if reader is not None:
            reader.join(timeout=1.0)

    analysis = analyze_poll_snapshots(snapshots, interval_seconds=interval_seconds, calendar=calendar)
    finalization_smoke: dict[str, Any] | None = None
    finalized_ledger_path: Path | None = None
    if cme_calendar_snapshot is not None and ledger_path.is_file():
        observations = load_observation_ledger(ledger_path)
        if observations:
            bar_starts = [int(row["bar_start_epoch_utc"]) for row in observations]
            window_start = datetime.fromtimestamp(min(bar_starts), timezone.utc)
            window_end = datetime.fromtimestamp(max(bar_starts) + 60, timezone.utc)
            try:
                schedule = IBKRContractSchedule([IBKRContractWindow(
                    con_id=con_id, local_symbol=local_symbol,
                    start_utc=window_start, end_utc=window_end + timedelta(minutes=1),
                )])
                finalized_ledger_path = target / f"ibkr_finalized_bars_{run_id}.jsonl"
                finalizer, finalized = finalize_observation_sequence(
                    observations, contract_schedule=schedule, calendar=calendar,
                    policy=FinalizationPolicy(),
                    finalization_ledger=AppendOnlyFinalizationLedger(finalized_ledger_path),
                )
                finalization_smoke = {
                    "scope": "single resolved contract window for diagnostic only; not a continuous roll schedule",
                    "calendar_version": calendar.version,
                    "contract_window_start_utc": window_start.isoformat(),
                    "contract_window_end_utc_exclusive": (window_end + timedelta(minutes=1)).isoformat(),
                    "observations_replayed_causally": len(observations),
                    "finalized_bars": len(finalized),
                    "first_finalized_bar_utc": finalized[0].exchange_bar_timestamp_utc.isoformat() if finalized else None,
                    "last_finalized_bar_utc": finalized[-1].exchange_bar_timestamp_utc.isoformat() if finalized else None,
                    "finalized_metadata_preserved": bool(finalized),
                    "late_revisions": finalizer.late_revisions,
                    "quality_events": finalizer.quality_events,
                    "policy": {"minimum_age_seconds": 600, "required_identical_observations": 2,
                               "minimum_observation_spacing_seconds": 30},
                    "finalization_ledger": str(finalized_ledger_path),
                }
            except Exception as exc:
                finalized_ledger_path = None
                finalization_smoke = {"error": f"{type(exc).__name__}: {exc}", "finalized_bars": 0}
    verdict = _poll_verdict(outcome, analysis, finalization_smoke)
    poll_summaries = [{key: value for key, value in item.items() if key != "bars"} | {
        "returned_bar_count": len(item.get("bars", [])),
        "first_bar_timestamp_utc": (datetime.fromtimestamp(min(int(b["timestamp"]) for b in item["bars"]), timezone.utc).isoformat()
                                     if item.get("bars") else None),
        "last_bar_timestamp_utc": (datetime.fromtimestamp(max(int(b["timestamp"]) for b in item["bars"]), timezone.utc).isoformat()
                                    if item.get("bars") else None),
    } for item in snapshots]
    report: dict[str, Any] = {
        "provider": "Interactive Brokers TWS/IB Gateway", "provenance": "TWS Demo (unverified by this diagnostic)",
        "mode_requested": "delayed (reqMarketDataType(3)); historical-bar entitlement/source not established by mode request",
        "endpoint": endpoint, "tws_reachable": reachable, "api_connected": api_connected,
        "contract_resolved": contract_resolved,
        "contract": contract_result, "bar_request": {"duration_seconds": duration_seconds, "bar_size": "1 min",
            "what_to_show": "TRADES", "use_rth": False, "format_date": 2, "completed_bars_only": True},
        "read_only": True, "orders_submitted": False, "run_started_utc": started_utc,
        "run_duration_seconds": time.monotonic() - started, "polls": poll_summaries, **analysis,
        "finalization_smoke": finalization_smoke,
        "provider_errors": app.all_errors, "failure": failure, "volume_caveat": "IBKR futures TRADES volume units are provider-specific; equivalence to Databento is unverified.",
        "provenance_caveat": "TWS Demo provenance is stated from operator context and is not cryptographically/provider-verified.",
        "latency_caveat": "Historical polling measures retrieval age at request/response time, not when IBKR first made a completed bar available.",
        "gap_caveat": "Calendar-classified closures are based on the supplied reviewed snapshot; expected-session OHLCV gaps remain unresolved and are not proof of data loss.",
        "verdict": verdict,
    }
    json_path = target / "ibkr_historical_poll.json"
    csv_path = target / "ibkr_historical_poll_bars.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["con_id", "timestamp", "timestamp_utc", "open", "high", "low", "close", "volume", "first_seen_utc", "last_seen_utc"])
        writer.writeheader()
        for row in analysis["completed_bars"]:
            writer.writerow({**{k: row[k] for k in ("con_id", "timestamp", "open", "high", "low", "close", "volume")},
                             "timestamp_utc": datetime.fromtimestamp(row["timestamp"], timezone.utc).isoformat(),
                             "first_seen_utc": row["first_seen_utc"], "last_seen_utc": row["last_seen_utc"]})
    report["report_paths"] = {"json": str(json_path), "csv": str(csv_path), "append_only_observations": str(ledger_path)}
    if finalized_ledger_path is not None:
        report["report_paths"]["append_only_finalized_bars"] = str(finalized_ledger_path)
    # Persist paths into JSON as well.
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7496, help="TWS Demo endpoint used for the bounded diagnostic")
    parser.add_argument("--client-id", type=int, default=74)
    parser.add_argument("--con-id", type=int, default=815824267)
    parser.add_argument("--local-symbol", default="MNQZ6")
    parser.add_argument("--polls", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output-dir", default="results/paper/ibkr_diagnostics/historical_poll")
    parser.add_argument("--calendar-snapshot", help="Optional reviewed CME calendar JSON to exercise delayed finalization in isolation")
    args = parser.parse_args()
    report = run_poll_test(host=args.host, port=args.port, client_id=args.client_id,
                           con_id=args.con_id, local_symbol=args.local_symbol, polls=args.polls,
                           interval_seconds=args.interval_seconds, duration_seconds=args.duration_seconds,
                           timeout=args.timeout, output_dir=args.output_dir,
                           cme_calendar_snapshot=args.calendar_snapshot)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"].startswith("historical_poll_usable") else 2


if __name__ == "__main__":
    raise SystemExit(main())
