"""Bounded, read-only smoke for cursor-aware delayed IBKR bar acquisition.

This diagnostic never calls an order API and does not connect to Paper Engine.
It is a single explicitly pinned MNQ contract, not a continuous roll feed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any

from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.ibkr_delayed_acquisition import (
    AppendOnlyGapClassificationLedger,
    AtomicAcquisitionCursor,
    DelayedAcquisitionSession,
    execute_with_bounded_reconnect,
    HistoricalRequestPacer,
    HistoricalRequestPlanner,
    format_ibkr_end_time,
    rehydrate_pending_from_observations,
)
from src.paper.ibkr_delayed_market_data import (
    AppendOnlyFinalizationLedger,
    DelayedBarFinalizer,
    FinalizationPolicy,
    IBKRContractSchedule,
    IBKRContractWindow,
)
from src.paper.ibkr_observation_ledger import AppendOnlyBarObservationLedger, load_observation_ledger


UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def run_smoke(*, host: str, port: int, client_id: int, con_id: int, local_symbol: str,
              expiry: str, cycles: int, interval_seconds: int, calendar_path: str | Path,
              output_dir: str | Path, timeout: float = 20.0) -> dict[str, Any]:
    if cycles < 3 or cycles > 5:
        raise ValueError("bounded smoke cycles must be 3..5")
    if interval_seconds < 30:
        raise ValueError("two-request polling cycles require at least 30 seconds")
    if con_id != 815824267 or local_symbol != "MNQZ6" or expiry != "20261218":
        raise ValueError("bounded validation is pinned to MNQZ6 / conId 815824267 / expiry 20261218")
    endpoint = f"{host}:{port}"
    try:
        with socket.create_connection((host, port), timeout=2):
            reachable = True
    except OSError as exc:
        return {"status": "unavailable", "endpoint": endpoint, "error": str(exc), "orders_submitted": False}

    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper

    class App(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.connected = threading.Event()
            self.contract_done = threading.Event()
            self.request_done = threading.Event()
            self.contracts = []
            self.rows = []
            self.request_errors = []
            self.all_errors = []
            self.active_req = None
            self.reader = None

        def nextValidId(self, orderId):
            self.connected.set()

        def contractDetails(self, reqId, details):
            if reqId == 1:
                self.contracts.append(details.contract)

        def contractDetailsEnd(self, reqId):
            if reqId == 1:
                self.contract_done.set()

        def historicalData(self, reqId, bar):
            if reqId != self.active_req:
                return
            try:
                self.rows.append({"con_id": con_id, "timestamp": int(bar.date),
                    "open": float(bar.open), "high": float(bar.high), "low": float(bar.low),
                    "close": float(bar.close), "volume": float(bar.volume),
                    "arrival_utc": _now().isoformat()})
            except (TypeError, ValueError, OverflowError) as exc:
                self.request_errors.append({"code": None, "message": f"malformed bar callback: {exc}"})

        def historicalDataEnd(self, reqId, start, end):
            if reqId == self.active_req:
                self.request_done.set()

        def error(self, reqId, code, message, *args):
            row = {"request_id": int(reqId), "code": int(code), "message": str(message),
                   "observed_at_utc": _now().isoformat()}
            self.all_errors.append(row)
            if reqId == self.active_req:
                self.request_errors.append(row)
                if int(code) in {162, 200, 321, 366}:
                    self.request_done.set()

        def connectionClosed(self):
            self.connected.clear()
            self.request_done.set()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    observation_path = target / "ibkr_observations.jsonl"
    delivery_path = target / "ibkr_finalized_bars.jsonl"
    cursor_path = target / "ibkr_acquisition_cursor.json"
    gap_path = target / "ibkr_gap_classifications.jsonl"
    cursor_store = AtomicAcquisitionCursor(cursor_path, contract_id=con_id,
                                           local_symbol=local_symbol, expiry=expiry)
    observation_ledger = AppendOnlyBarObservationLedger(observation_path, run_id="ibkr-delayed-smoke")
    delivery_ledger = AppendOnlyFinalizationLedger(delivery_path)
    gap_ledger = AppendOnlyGapClassificationLedger(gap_path)
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(calendar_path))
    schedule = IBKRContractSchedule([IBKRContractWindow(
        con_id, local_symbol,
        datetime.combine(calendar.snapshot.coverage_start - timedelta(days=1), datetime.min.time(), UTC),
        datetime.combine(calendar.snapshot.coverage_end + timedelta(days=2), datetime.min.time(), UTC),
    )])
    finalizer = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
        policy=FinalizationPolicy(), finalization_ledger=delivery_ledger,
        gap_classifications=gap_ledger.rows)
    rehydrate_pending_from_observations(finalizer, load_observation_ledger(observation_path))
    session = DelayedAcquisitionSession(cursor_store=cursor_store,
                                         observation_ledger=observation_ledger, finalizer=finalizer)
    session.pending_confirmation_starts()
    prior_observations = load_observation_ledger(observation_path)
    poll_number_offset = max((int(row["poll_number"]) for row in prior_observations), default=0)
    planner = HistoricalRequestPlanner()
    pacer = HistoricalRequestPacer(prior_request_epochs=cursor_store.cursor.recent_request_epochs)
    app = App()
    reader = None
    snapshots = []
    errors = []
    emitted = []
    max_request_count = 0
    reconnect_count = 0
    actual_requests_issued = 0
    start_monotonic = time.monotonic()
    contract_info = None
    try:
        app.connect(host, port, client_id)
        reader = threading.Thread(target=app.run, name="ibkr-delayed-feed-smoke", daemon=True)
        reader.start()
        if not app.connected.wait(timeout):
            raise RuntimeError("TWS API handshake timed out")
        query = Contract()
        query.conId = con_id
        query.exchange = "CME"
        app.reqContractDetails(1, query)
        if not app.contract_done.wait(timeout):
            raise RuntimeError("pinned MNQ contract lookup timed out")
        matches = [item for item in app.contracts if int(item.conId) == con_id]
        if len(matches) != 1:
            raise RuntimeError(f"expected one contract with conId={con_id}; got {len(matches)}")
        contract = matches[0]
        if (contract.symbol != "MNQ" or contract.secType != "FUT" or
                contract.localSymbol != local_symbol or
                str(contract.lastTradeDateOrContractMonth) != expiry):
            raise RuntimeError("resolved contract identity/expiry differs from pinned MNQZ6")
        contract_info = {"symbol": contract.symbol, "local_symbol": contract.localSymbol,
                         "con_id": int(contract.conId), "expiry": contract.lastTradeDateOrContractMonth,
                         "exchange": contract.exchange}
        app.reqMarketDataType(3)

        def reconnect_and_resolve():
            nonlocal app, reader, contract, contract_info, reconnect_count
            old_app, old_reader = app, reader
            if old_app.isConnected():
                old_app.disconnect()
            if old_reader is not None:
                old_reader.join(timeout=2)
            app = App()
            app.connect(host, port, client_id)
            reader = threading.Thread(target=app.run, name="ibkr-delayed-reconnect", daemon=True)
            reader.start()
            if not app.connected.wait(timeout):
                raise RuntimeError("TWS reconnect handshake timed out")
            query = Contract()
            query.conId = con_id
            query.exchange = "CME"
            app.reqContractDetails(1, query)
            if not app.contract_done.wait(timeout):
                raise RuntimeError("contract re-resolution timed out after reconnect")
            matches = [item for item in app.contracts if int(item.conId) == con_id]
            if len(matches) != 1:
                raise RuntimeError("pinned MNQ contract did not resolve uniquely after reconnect")
            contract = matches[0]
            if (contract.symbol != "MNQ" or contract.secType != "FUT" or
                    contract.localSymbol != local_symbol or
                    str(contract.lastTradeDateOrContractMonth) != expiry):
                raise RuntimeError("contract identity changed during TWS reconnect")
            contract_info = {"symbol": contract.symbol, "local_symbol": contract.localSymbol,
                "con_id": int(contract.conId), "expiry": contract.lastTradeDateOrContractMonth,
                "exchange": contract.exchange}
            app.reqMarketDataType(3)
            reconnect_count += 1
        last_cycle_started = None
        next_req_id = 10
        prior_epochs = pacer.recent_request_epochs
        if prior_epochs:
            wait_for_pacing = pacer.minimum_cycle_seconds - (_now().timestamp() - max(prior_epochs))
            if wait_for_pacing > 0:
                time.sleep(wait_for_pacing + 1.0)
        for cycle in range(1, cycles + 1):
            if last_cycle_started is not None:
                delay = interval_seconds - (time.monotonic() - last_cycle_started)
                if delay > 0:
                    time.sleep(delay + 1.0)
            last_cycle_started = time.monotonic()
            now = _now()
            requests = planner.plan_cycle(now_utc=now,
                pending_bar_starts=session.pending_confirmation_starts(), include_discovery=True,
                confirmation_offset=cursor_store.cursor.pending_confirmation_offset)
            confirmation = next((item for item in requests if item.kind == "pending_confirmation"), None)
            if confirmation is not None and confirmation.confirmation_group_count:
                cursor_store.cursor.pending_confirmation_offset = (
                    confirmation.confirmation_group_index + 1) % confirmation.confirmation_group_count
            max_request_count = max(max_request_count, len(requests))
            pacer.begin_cycle(now, len(requests))
            session.persist_request_budget(pacer, now=now)
            cycle_rows = 0
            for request in requests:
                if request.kind == "recent_discovery":
                    end_time, duration = "", f"{request.duration_seconds} S"
                else:
                    end_time = format_ibkr_end_time(request.end_time_utc)
                    duration = f"{request.duration_seconds} S"
                retry_state = {"request_id": next_req_id}

                def attempt_request():
                    nonlocal actual_requests_issued
                    app.rows = []
                    app.request_errors = []
                    app.request_done.clear()
                    req_id = retry_state["request_id"]
                    app.active_req = req_id
                    actual_requests_issued += 1
                    app.reqHistoricalData(req_id, contract, end_time, duration, "1 min", "TRADES", 0, 2, False, [])
                    done = app.request_done.wait(timeout)
                    if not done:
                        try:
                            app.cancelHistoricalData(req_id)
                        except Exception:
                            pass
                    return {"completed": done and not app.request_errors,
                            "connected": app.isConnected(), "done": done,
                            "rows": list(app.rows), "errors": list(app.request_errors),
                            "request_id": req_id}

                def reconnect_transport():
                    nonlocal next_req_id
                    reconnect_and_resolve()
                    next_req_id += 1
                    retry_state["request_id"] = next_req_id
                    pacer.reserve_reconnect_retry(_now())
                    session.persist_request_budget(pacer, now=_now())

                result, reconnects = execute_with_bounded_reconnect(
                    attempt_request, reconnect_transport, max_reconnects=1)
                reconnect_count += reconnects
                if not result.get("completed"):
                    if not result.get("done"):
                        errors.append({"cycle": cycle, "request_id": result["request_id"],
                                       "kind": request.kind, "error": "historical request timed out"})
                    else:
                        errors.extend({"cycle": cycle, "kind": request.kind, **item}
                                      for item in result.get("errors", []))
                    break
                rows = sorted(result["rows"], key=lambda item: int(item["timestamp"]))
                arrival = max((datetime.fromisoformat(item["arrival_utc"]) for item in rows), default=_now())
                delivered_now = session.ingest_response(rows, observed_at=arrival,
                    poll_number=poll_number_offset + cycle, request_id=next_req_id,
                    finalize_timestamps=(request.covered_pending_starts
                                         if request.kind == "pending_confirmation" else None))
                emitted.extend(delivered_now)
                cycle_rows += len(rows)
                snapshots.append({"cycle": cycle, "request_id": result["request_id"], "kind": request.kind,
                    "duration_seconds": request.duration_seconds,
                    "end_time_utc": request.end_time_utc.isoformat() if request.end_time_utc else None,
                    "pending_count_after_response": len(session.pending_confirmation_starts()),
                    "rows_returned": len(rows), "finalized_this_response": len(delivered_now)})
                next_req_id = max(next_req_id, result["request_id"]) + 1
            if errors:
                break
    except Exception as exc:
        errors.append({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if app.isConnected():
            app.disconnect()
        if reader is not None:
            reader.join(timeout=1)

    late_revisions = list(finalizer.late_revisions)
    delays = [(item.finalized_at_utc - (item.exchange_bar_timestamp_utc + timedelta(minutes=1))).total_seconds()
              for item in emitted]
    report = {
        "status": "completed" if contract_info is not None and not errors else "incomplete",
        "provider": "IBKR TWS historical TRADES", "provenance": "TWS Demo unverified",
        "delayed_mode_requested": 3, "orders_submitted": False, "paper_engine_connected": False,
        "pinned_contract": contract_info, "calendar_version": calendar.version,
        "cycles_requested": cycles, "poll_interval_seconds": interval_seconds,
        "requests_issued": actual_requests_issued, "max_planned_requests_per_cycle": max_request_count,
        "reconnect_count": reconnect_count,
        "requests_detail": snapshots,
        "observations_appended_total": len(load_observation_ledger(observation_path)),
        "observed_unique_bars": len({(int(row["contract_id"]), int(row["bar_start_epoch_utc"]))
                                      for row in load_observation_ledger(observation_path)}),
        "finalized_bars_emitted_this_run": len(emitted),
        "first_emitted_utc": emitted[0].exchange_bar_timestamp_utc.isoformat() if emitted else None,
        "last_emitted_utc": emitted[-1].exchange_bar_timestamp_utc.isoformat() if emitted else None,
        "oldest_pending_utc": finalizer.pending_timestamps[0].isoformat() if finalizer.pending_timestamps else None,
        "measured_finalization_delay_seconds": {"min": min(delays) if delays else None,
            "median": sorted(delays)[len(delays)//2] if delays else None,
            "max": max(delays) if delays else None,
            "basis": "client finalization time minus exchange bar end; not provider publication latency"},
        "pre_finalization_revisions": [row for row in finalizer.quality_events
                                        if row.get("kind") == "pre_finalization_revision"],
        "late_revisions": late_revisions,
        "gaps": finalizer.quality_events,
        "provider_errors": app.all_errors, "errors": errors,
        "runtime_seconds": time.monotonic() - start_monotonic,
        "state_files": {"observations": str(observation_path), "finalized": str(delivery_path),
                        "cursor": str(cursor_path), "gap_classifications": str(gap_path)},
        "limitations": ["single pinned expiry only; no continuous roll schedule",
            "historical-bar entitlement/provenance is not verified by reqMarketDataType(3)",
            "not connected to Paper Engine; no order API used"],
    }
    report_path = target / "ibkr_delayed_feed_smoke.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=76)
    parser.add_argument("--con-id", type=int, default=815824267)
    parser.add_argument("--local-symbol", default="MNQZ6")
    parser.add_argument("--expiry", default="20261218")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--calendar-snapshot", required=True)
    parser.add_argument("--output-dir", default="results/paper/ibkr_diagnostics/cursor_aware_smoke")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    report = run_smoke(host=args.host, port=args.port, client_id=args.client_id,
        con_id=args.con_id, local_symbol=args.local_symbol, expiry=args.expiry,
        cycles=args.cycles, interval_seconds=args.interval_seconds,
        calendar_path=args.calendar_snapshot, output_dir=args.output_dir, timeout=args.timeout)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" and not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
