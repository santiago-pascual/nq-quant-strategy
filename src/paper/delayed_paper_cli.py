"""Fail-closed lifecycle CLI for strategy-enabled delayed IBKR Paper.

The ``readiness`` command is offline and never starts TWS or Paper. ``start``
is available only after an independently produced recovery-validation record,
reviewed calendar, compatible causal bootstrap checkpoint and explicit
operator confirmation are supplied. No broker order APIs are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALENDAR = ROOT / "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json"
DEFAULT_REVIEW = ROOT / "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.review.json"
DEFAULT_RECOVERY_VALIDATION = ROOT / "results/diagnostics/delayed_paper_strategy_recovery_validation_20261009.json"
EXPECTED_CON_ID = 815824267
EXPECTED_LOCAL_SYMBOL = "MNQZ6"
EXPECTED_EXPIRY = "20261218"


def _expected_risk_configuration() -> dict[str, Any]:
    from src.risk.policy import XFA_50K_PRODUCTION_POLICY
    return asdict(XFA_50K_PRODUCTION_POLICY)


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("activation timestamp must include a timezone")
    return stamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readiness_report(*, activation: str, calendar_path: Path | None,
                     review_path: Path | None, bootstrap_path: Path | None,
                     identity_path: Path | None, recovery_validation_path: Path | None,
                     cost_config: Path | None, output_dir: Path | None,
                     host: str | None, port: int | None) -> dict[str, Any]:
    blockers: list[str] = []
    checks: dict[str, Any] = {"mode": "DELAYED_IBKR_PAPER", "orders_enabled": False}
    try:
        activation_ts = _utc(activation)
        checks["activation_timestamp_utc"] = activation_ts.isoformat()
    except (ValueError, TypeError) as exc:
        activation_ts = None
        blockers.append(f"invalid activation timestamp: {exc}")

    calendar = None
    calendar_reviewed = False
    if not calendar_path or not calendar_path.is_file():
        blockers.append("reviewed CME calendar snapshot is missing")
    else:
        try:
            from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
            calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(calendar_path))
            review = json.loads(review_path.read_text(encoding="utf-8")) if review_path and review_path.is_file() else {}
            if (review.get("snapshot_identity") != calendar.snapshot.identity
                    or review.get("product") != "MNQ"
                    or review.get("venue") != "CME Globex"
                    or not str(review.get("review_status", "")).startswith("REVIEWED")):
                blockers.append("calendar review record is absent or does not certify this MNQ snapshot")
            else:
                calendar_reviewed = True
            if activation_ts is not None:
                trade_date = calendar.trading_date_for_timestamp(activation_ts)
                if trade_date is None:
                    blockers.append("activation timestamp is outside a verified open Globex minute")
                elif not calendar.snapshot.coverage_start <= trade_date <= calendar.snapshot.coverage_end:
                    blockers.append("CME calendar does not cover the activation trade date")
            checks["calendar"] = {
                "version": calendar.version,
                "identity": calendar.snapshot.identity,
                "coverage_start": calendar.snapshot.coverage_start.isoformat(),
                "coverage_end": calendar.snapshot.coverage_end.isoformat(),
                "reviewed": calendar_reviewed,
            }
        except Exception as exc:
            blockers.append(f"CME calendar validation failed: {type(exc).__name__}: {exc}")

    artifact = None
    if not bootstrap_path or not bootstrap_path.is_file():
        blockers.append("activation-ready causal bootstrap artifact is missing")
    elif not identity_path or not identity_path.is_file():
        blockers.append("trusted expected bootstrap identity manifest is missing")
    else:
        try:
            from src.paper.causal_bootstrap import (
                CausalBootstrapCheckpointStore,
                activation_payload_errors,
            )
            expected_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            artifact = CausalBootstrapCheckpointStore(bootstrap_path).load(
                expected_identity=expected_identity
            )
            if activation_ts is not None:
                blockers.extend(
                    "bootstrap activation schema: " + error
                    for error in activation_payload_errors(
                        artifact, activation_timestamp=activation_ts,
                        expected_identity=expected_identity,
                    )
                )
            if not artifact.get("ready_for_activation"):
                blockers.append("bootstrap artifact is not marked ready_for_activation")
            strategy_seed = artifact.get("strategy_state_seed", {})
            if (strategy_seed.get("schema_version") != 1
                    or strategy_seed.get("initialization") != "fresh_strategy_constructors_at_activation"
                    or set(strategy_seed.get("strategies", {})) != {"MRL1", "MRS2", "S2R", "ORB"}
                    or strategy_seed.get("no_historical_strategy_execution") is not True):
                blockers.append("bootstrap artifact lacks the complete fresh four-strategy state seed")
            account_seed = artifact.get("account_seed", {})
            if (not isinstance(account_seed.get("initial_equity"), (int, float))
                    or account_seed.get("risk_configuration") is None
                    or any(account_seed.get(name) != [] for name in ("positions", "orders", "trades"))):
                blockers.append("bootstrap artifact lacks a valid fresh simulated-account seed")
            if activation_ts is not None and pd.Timestamp(artifact["activation_timestamp_utc"]) != activation_ts:
                blockers.append("bootstrap artifact activation timestamp does not match the requested activation")
            checks["bootstrap"] = {
                "identity_match": True,
                "ready_for_activation": bool(artifact.get("ready_for_activation")),
                "last_processed_timestamp_utc": artifact.get("last_processed_timestamp_utc"),
                "artifact_sha256": _sha256(bootstrap_path),
            }
        except Exception as exc:
            blockers.append(f"bootstrap artifact validation failed: {type(exc).__name__}: {exc}")

    if not recovery_validation_path or not recovery_validation_path.is_file():
        blockers.append("strategy-enabled full-state crash-recovery validation record is missing")
    else:
        try:
            evidence = json.loads(recovery_validation_path.read_text(encoding="utf-8"))
            from src.paper.realtime_checkpoint import runtime_identity
            from src.paper.runtime_compatibility import validate_runtime_identity
            fingerprint_match = validate_runtime_identity(
                evidence.get("runtime_fingerprint"), runtime_identity()
            ).accepted
            test_sources = evidence.get("test_sources", {})
            test_sources_match = bool(test_sources) and all(
                (ROOT / relative).is_file() and _sha256(ROOT / relative) == digest
                for relative, digest in test_sources.items()
            )
            required = (evidence.get("status") == "PASSED"
                        and evidence.get("all_four_strategies") is True
                        and evidence.get("fitted_hmm_and_refit_boundary") is True
                        and evidence.get("one_hour_catchup") is True
                        and evidence.get("uninterrupted_equals_recovered") is True
                        and evidence.get("pending_strategy_order_checkpoint_restore") is True
                        and fingerprint_match and test_sources_match)
            if not required:
                blockers.append(
                    "recovery validation record is incomplete or its code/test fingerprint is stale"
                )
            checks["recovery_validation"] = {
                "passed": bool(required),
                "path": str(recovery_validation_path.resolve()),
                "runtime_fingerprint_match": fingerprint_match,
                "test_sources_match": test_sources_match,
            }
        except Exception as exc:
            blockers.append(f"recovery validation record cannot be read: {type(exc).__name__}: {exc}")

    if not cost_config or not cost_config.is_file():
        blockers.append("explicit MNQ Paper cost profile is missing")
    else:
        try:
            from src.paper.costs import PaperCostPolicy
            policy = PaperCostPolicy.from_json(cost_config)
            if policy.symbol != "MNQ" or policy.artificial_slippage_ticks != 0:
                blockers.append("cost profile is incompatible with delayed MNQ Paper")
            checks["cost_profile"] = {"profile_id": policy.profile_id, "identity": policy.identity}
        except Exception as exc:
            blockers.append(f"cost profile validation failed: {type(exc).__name__}: {exc}")

    if not output_dir:
        blockers.append("dedicated delayed-Paper output directory is required")
    else:
        resolved = output_dir.resolve()
        if not str(resolved).lower().startswith(str((ROOT / "results/paper/delayed_").resolve()).lower()):
            blockers.append("output directory must be a dedicated results/paper/delayed_* path")
        checks["output_directory"] = str(resolved)

    checks["contract"] = {"symbol": "MNQ", "local_symbol": EXPECTED_LOCAL_SYMBOL,
                           "con_id": EXPECTED_CON_ID, "expiry": EXPECTED_EXPIRY,
                           "roll_policy": "fail closed; no schedule beyond approved MNQZ6 mapping"}
    checks["ibapi_available"] = importlib.util.find_spec("ibapi") is not None
    if not checks["ibapi_available"]:
        blockers.append("IBKR API package is unavailable in the selected Python environment")
    checks["tws_endpoint_configured"] = bool(host and port)
    if not host or not port:
        blockers.append("TWS host and port must be explicitly configured")
    checks["bootstrap_payload_present"] = artifact is not None
    checks["calendar_valid"] = calendar is not None
    checks["ready"] = not blockers
    checks["blockers"] = blockers
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNQ delayed IBKR PAPER service (simulated fills only).")
    parser.add_argument("command", choices=("readiness", "start", "stop", "status"))
    parser.add_argument("--activation-utc", default="2026-10-09T00:00:00Z")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--calendar-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--bootstrap-identity", type=Path)
    parser.add_argument("--recovery-validation", type=Path, default=DEFAULT_RECOVERY_VALIDATION)
    parser.add_argument("--cost-config", type=Path, default=ROOT / "src/paper/config/topstepx_mnq_fees_2026-07.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tws-host", default="127.0.0.1")
    parser.add_argument("--tws-port", type=int, default=7496)
    parser.add_argument("--tws-client-id", type=int, default=93)
    parser.add_argument("--confirm-delayed-paper", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir
    paper_root = (ROOT / "results/paper").resolve()
    try:
        relative_output = output.resolve().relative_to(paper_root)
    except ValueError:
        relative_output = Path()
    is_delayed_output = bool(relative_output.parts and relative_output.parts[0].startswith("delayed_"))
    if args.command == "stop":
        manifest_path = output / "delayed_paper_run.json"
        if not is_delayed_output or not manifest_path.is_file():
            print(json.dumps({"stop_requested": False,
                              "reason": "target is not a known delayed-Paper run directory"}, indent=2))
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("mode") != "DELAYED_IBKR_PAPER":
            print(json.dumps({"stop_requested": False, "reason": "run manifest mode mismatch"}, indent=2))
            return 2
        status_path = output / "status.json"
        if not status_path.exists():
            print(json.dumps({"stopped": False, "reason": "no Paper status file found"}))
            return 2
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = status.get("system", {}).get("state")
        if state not in {"RUNNING", "DEGRADED", "PAUSED_REFIT"}:
            print(json.dumps({"stop_requested": False, "reason": f"Paper is not active (state={state})"}, indent=2))
            return 2
        (output / "stop.request").write_text("operator requested graceful stop\n", encoding="utf-8")
        print(json.dumps({"stop_requested": True, "mode": "DELAYED_IBKR_PAPER",
                          "output_dir": str(output.resolve())}, indent=2))
        return 0
    if args.command == "status":
        manifest_path = output / "delayed_paper_run.json"
        if not is_delayed_output or not manifest_path.is_file():
            print(json.dumps({"available": False, "mode": "DELAYED_IBKR_PAPER",
                              "reason": "not a known delayed-Paper run directory"}, indent=2))
            return 2
        path = output / "status.json"
        if not path.is_file():
            print(json.dumps({"available": False, "mode": "DELAYED_IBKR_PAPER"}, indent=2))
            return 2
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({"available": True, "paper_engine_mode": payload.get("mode"),
                          "feed_mode": "DELAYED_IBKR_PAPER", **payload}, indent=2, default=str))
        return 0

    report = readiness_report(
        activation=args.activation_utc, calendar_path=args.calendar,
        review_path=args.calendar_review, bootstrap_path=args.bootstrap,
        identity_path=args.bootstrap_identity,
        recovery_validation_path=args.recovery_validation,
        cost_config=args.cost_config, output_dir=output,
        host=args.tws_host, port=args.tws_port,
    )
    if args.command == "readiness":
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ready"] else 2
    if not args.confirm_delayed_paper:
        report["blockers"].append("start requires --confirm-delayed-paper")
        print(json.dumps(report, indent=2, default=str))
        return 2
    if args.resume:
        manifest = output / "delayed_paper_run.json"
        checkpoint = output / "paper_checkpoint.json"
        if not manifest.is_file() or not checkpoint.is_file():
            report["blockers"].append("resume requires this delayed-Paper run manifest and Paper checkpoint")
    elif output.exists() and any(output.iterdir()):
        report["blockers"].append("new delayed-Paper output directory is not empty")
    if report["blockers"]:
        print(json.dumps(report, indent=2, default=str))
        return 2

    # All prerequisites are validated before any TWS connection or Paper write.
    from src.paper.causal_bootstrap import CausalBootstrapCheckpointStore, load_context_seed_for_new_account
    from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
    from src.paper.costs import PaperCostPolicy
    from src.paper.ibkr_paper_runner import TWSHistoricalTRADESClient, build_cursor_aware_pipeline
    from src.paper.ibkr_paper_recovery import AcknowledgedDelayedPaperService
    from src.paper.realtime_service import RealtimePaperConfig
    from src.paper.run_autonomous import build_real_paper_engine

    output.mkdir(parents=True, exist_ok=True)
    activation = _utc(args.activation_utc)
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(args.calendar))
    costs = PaperCostPolicy.from_json(args.cost_config)
    expected_identity = json.loads(args.bootstrap_identity.read_text(encoding="utf-8"))
    checkpoint_store = CausalBootstrapCheckpointStore(args.bootstrap)
    bootstrap = checkpoint_store.load(expected_identity=expected_identity)
    initial_equity = float(bootstrap["account_seed"]["initial_equity"])
    expected_risk = _expected_risk_configuration()
    if bootstrap["account_seed"].get("risk_configuration") != expected_risk:
        raise SystemExit("Bootstrap risk configuration does not match frozen Paper risk policy")
    engine, adapter = build_real_paper_engine(
        output, initial_equity=initial_equity,
        commission_per_contract=costs.commission_per_contract_side,
        exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
        regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
        recover_trailing_event=args.resume,
    )
    if args.resume:
        saved = json.loads((output / "delayed_paper_run.json").read_text(encoding="utf-8"))
        if saved.get("activation_timestamp_utc") != activation.isoformat():
            raise SystemExit("Resume activation timestamp differs from saved delayed Paper manifest")
    else:
        load_context_seed_for_new_account(
            store=checkpoint_store, context=adapter.context,
            expected_identity=expected_identity, activation_timestamp=activation,
        )
        (output / "delayed_paper_run.json").write_text(json.dumps({
            "mode": "DELAYED_IBKR_PAPER", "paper_only": True,
            "orders_enabled": False, "activation_timestamp_utc": activation.isoformat(),
            "contract": {"con_id": EXPECTED_CON_ID, "local_symbol": EXPECTED_LOCAL_SYMBOL,
                         "expiry": EXPECTED_EXPIRY},
            "calendar_identity": calendar.snapshot.identity,
            "bootstrap_identity": expected_identity,
            "bootstrap_sha256": _sha256(args.bootstrap),
        }, indent=2) + "\n", encoding="utf-8")
    client = TWSHistoricalTRADESClient(
        host=args.tws_host, port=args.tws_port, client_id=args.tws_client_id,
        con_id=EXPECTED_CON_ID, local_symbol=EXPECTED_LOCAL_SYMBOL, expiry=EXPECTED_EXPIRY,
    )
    try:
        pipeline = build_cursor_aware_pipeline(
            output_dir=output, calendar=calendar,
            request_historical=client.request, reconnect=client.reconnect,
        )
        pipeline["source"].bootstrap_after_timestamp = activation.to_pydatetime() - timedelta(minutes=1)
        service = AcknowledgedDelayedPaperService(
            source=pipeline["source"], engine=engine, context_adapter=adapter,
            config=RealtimePaperConfig(mode="PAPER", output_dir=output, checkpoint_every_bars=1),
            calendar=calendar, cost_policy=costs,
            delivery_ledger=pipeline["delivery_ledger"], run_id="delayed-ibkr-paper",
        )
        from src.paper.run_realtime_paper import _run_with_graceful_signals
        _run_with_graceful_signals(service, restore=args.resume)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
